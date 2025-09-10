"""
PennAir Challenge - Dynamic Shape Detection and Tracking
=======================================================

This module processes video streams to detect and track colored shapes on different backgrounds.
It combines color-based detection (LAB color space) with motion detection (MOG2) for
shape identification and provides both 2D pixel coordinates and 3D world coordinates.

Key Features:
- Dual detection: Color-based (LAB distance) + Motion-based (background subtraction)
- Tunable parameters: Gaussian blur, threshold values, morphological operations
- 3D coordinate estimation using pinhole camera model and reference circle
- Real-time processing with optional live preview
"""

import argparse
import os
from typing import List, Tuple, Optional

import cv2
import numpy as np

# Import helper functions from static image detector module
from detect_shapes import (
	compute_background_lab,     # Estimates grass background color from image borders
	build_foreground_mask,      # Creates binary mask for non-grass regions
	find_contours,              # Finds and filters contours by area
	classify_shape,             # Classifies contours as circle, triangle, square, etc.
	essential_kernel            # 5x5 morphological kernel for noise reduction
)

# === CAMERA CALIBRATION PARAMETERS ===
# These values are specific to the camera from the provided intrinsic matrix
DEFAULT_FX = 2564.3186869      # Focal length in x-direction (pixels)
DEFAULT_FY = 2569.70273111     # Focal length in y-direction (pixels)
DEFAULT_CIRCLE_RADIUS_IN = 10.0  # Known real-world radius of reference circle (inches)

# === 3D CALIBRATION CACHE ===
# These values are computed once from the reference circle and reused for all frames
# This assumes constant depth (planar scene) which is valid for objects on flat plane
Z0_CACHE: Optional[float] = None        # Constant depth from camera to ground plane
ALPHA_X_CACHE: Optional[float] = None   # Pre-computed scale factor: Z0 / fx
ALPHA_Y_CACHE: Optional[float] = None   # Pre-computed scale factor: Z0 / fy



def process_video_frame(
    frame: np.ndarray,
    background_lab: Optional[np.ndarray] = None,
    mog2: Optional[cv2.BackgroundSubtractor] = None,
    *,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cx: float = 0.0,
    cy: float = 0.0,
    circle_radius_in: Optional[float] = None,
    label_3d: bool = False,
    blur_kernel_size: int = 5,
    blur_sigma: float = 0.0,
    threshold: int = 95
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Process a single video frame for shape detection using dual-method approach.
    
    This function combines color-based detection (LAB color space distance from grass)
    with motion-based detection (MOG2 background subtraction) to identify shapes on different backgrounds.
    
    Args:
        frame: Input BGR frame from video
        background_lab: Pre-computed grass background color in LAB space
        mog2: MOG2 background subtractor for motion detection
        fx, fy: Camera focal lengths for 3D coordinate calculation
        cx, cy: Image center coordinates (origin for coordinate system)
        circle_radius_in: Known real-world radius of reference circle
        label_3d: Whether to compute and display 3D coordinates
        blur_kernel_size: Gaussian blur kernel size (reduces noise)
        blur_sigma: Gaussian blur strength (0 = auto-calculate)
        threshold: LAB distance threshold for foreground detection
        
    Returns:
        Tuple of (annotated_frame, detected_contours, combined_mask)
    """
    # Fallback: compute background from current frame if not provided
    if background_lab is None:
        print("No background provided, computing from frame")
        background_lab = compute_background_lab(frame)

    # === COLOR-BASED DETECTION ===
    # Create mask based on LAB color distance from background
    lab_mask = build_foreground_mask_with_background(frame, background_lab, blur_kernel_size, blur_sigma, threshold)

    # === MOTION-BASED DETECTION ===
    # Use MOG2 to detect moving objects that cannot be detected with color mask
    motion_mask = None
    if mog2 is not None:
        fg = mog2.apply(frame)
        # Remove shadows (value 127) and keep only strong foreground (255)
        _, motion_mask = cv2.threshold(fg, 128, 255, cv2.THRESH_BINARY)
        # Clean up motion mask with morphological operations
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, essential_kernel, iterations=1)   # Remove noise
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, essential_kernel, iterations=1)  # Fill gaps
    else:
        motion_mask = np.zeros_like(lab_mask)

    # === MASK FUSION ===
    # Combine both detection methods: pixel is foreground if EITHER method detects it
    # cv2.bitwise_or performs a pixel-wise logical OR operation between the two masks.
    # For each pixel location, if either lab_mask or motion_mask is nonzero (i.e., foreground),
    # the output combined_mask will be set to 255 (foreground) at that pixel.
    # This merges the results so that any pixel detected by either method is included in the final mask.
    combined_mask = cv2.bitwise_or(lab_mask, motion_mask)

    # === CONTOUR DETECTION ===
    # Find shapes and filter by area to remove noise and overly large regions
    contours = find_contours_with_min_area(combined_mask, min_area=10000, max_area=150000)

    # === ANNOTATION ===
    # Draw detected shapes with labels and coordinate information
    annotated_frame = annotate_video_frame(
        frame, contours,
        fx=fx, fy=fy, cx=cx, cy=cy,
        circle_radius_in=circle_radius_in,
        label_3d=label_3d
    )

    return annotated_frame, contours, combined_mask


def _estimate_depth_from_circle(contours: List[np.ndarray], fx: float, fy: float, circle_radius_in: float):
    """
    Find a circle among contours and calculate depth Z (in same units as circle_radius_in).
    Uses the minEnclosingCircle radius in pixels and pinhole model: Z = f * R_real / r_pix.
    Returns (Z, r_pix) or (None, None) if no circle found.
    """
    if fx is None or fy is None or circle_radius_in is None:
        return None, None

    f = 0.5 * (float(fx) + float(fy))
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        name = classify_shape(cnt).lower()
        if "circle" in name:
            (cx, cy), r_pix = cv2.minEnclosingCircle(cnt)
            if r_pix > 5:
                Z = (f * float(circle_radius_in)) / float(r_pix)
                return float(Z), float(r_pix)
    return None, None


def _pixel_to_camera_xyz(u: float, v: float, fx: float, fy: float, cx: float, cy: float, Z: float):
    """
    Back-project pixel (u,v) to camera frame at depth Z.
    X = (u - cx) * Z / fx ;  Y = (v - cy) * Z / fy ;  Z = Z
    """
    X = (float(u) - float(cx)) * float(Z) / float(fx)
    Y = (float(v) - float(cy)) * float(Z) / float(fy)
    return float(X), float(Y), float(Z)


def find_contours_with_min_area(mask: np.ndarray, min_area: int = 50, max_area: int = 150000) -> List[np.ndarray]:
	"""Find contours with a minimum area filter to remove noise."""
    #finds the contours of the shapes in the mask that are external (ignores holes and nested contours). 
	#cv2.RETR_EXTERNAL is used to find the external contours only.
	#cv2.CHAIN_APPROX_SIMPLE is used to only return the corners of the contours, saving memory, increasing efficiency
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# Filter contours by minimum area
	large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area and cv2.contourArea(cnt) <= max_area]
	# sorts the contours by the x-coordinate of the bounding rectangle of the contour
	#cv2.boundingRect(c)[0] extracts the x-coordinate of the bounding rectangle of each contour
	#which is used to sort them left to right.
	large_contours.sort(key=lambda c: cv2.boundingRect(c)[0])
	return large_contours


def build_foreground_mask_with_background(
    image_bgr: np.ndarray, 
    background_lab: np.ndarray,
    blur_kernel_size: int = 5,
    blur_sigma: float = 0.0,
    threshold: int = 95
) -> np.ndarray:
	"""
	Create foreground mask using LAB color space distance from grass background.
	
	This is the core color-based detection algorithm. It works by:
	1. Blurring the image to reduce texture noise from grass
	2. Converting to LAB color space (better for color distance calculations)
	3. Computing Euclidean distance from each pixel to the grass background color
	4. Thresholding to create binary mask (foreground vs background)
	5. Morphological operations to clean up the mask
	
	Args:
		image_bgr: Input frame in BGR color space
		background_lab: Pre-computed grass background color in LAB space
		blur_kernel_size: Size of Gaussian blur kernel (must be odd)
		blur_sigma: Gaussian blur standard deviation (0 = auto-calculate)
		threshold: Distance threshold for foreground detection (0-255)
		
	Returns:
		Binary mask where 255 = foreground (shapes), 0 = background (grass)
	"""
	# Ensure kernel size is odd (required by OpenCV)
	if blur_kernel_size % 2 == 0:
		blur_kernel_size += 1
	
	# Apply Gaussian blur to reduce grass texture noise
	blurred = cv2.GaussianBlur(image_bgr, (blur_kernel_size, blur_kernel_size), blur_sigma)
	
	# Convert to LAB color space for perceptually uniform color distance
	lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)

	# Compute Euclidean distance from each pixel to grass background color
	# The background_lab is a 1D array of shape (3,) representing the LAB color of the background.
	# To subtract it from every pixel in the image (which has shape (H, W, 3)), we need to broadcast it.
	# The reshape(1, 1, 3) turns background_lab into a (1, 1, 3) array, so that when we subtract it from
	# the (H, W, 3) lab image, NumPy broadcasts the background color across all pixels.
	delta = np.linalg.norm(lab - background_lab.reshape(1, 1, 3), axis=2)
	
	# Normalize distance values to 0-255 range for thresholding
	delta_norm = cv2.normalize(delta, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

	#applies a threshold to the normalized distance to create a binary mask
	#A mask is a binary image that where pictures where the color difference is greater 
	# than the threshold are white, and pictures where the color difference is less than the threshold are black
	_, mask = cv2.threshold(delta_norm, threshold, 255, cv2.THRESH_BINARY)

	#applies a "dilation", then "erosion" to the mask to fill small holes inside detected shapes and joins nearby white regions
	#FILLS HOLES
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, essential_kernel, iterations=2)  # Fill small holes

    #applies a "erosion", then "dilation" to the mask to remove small blobs left in the grass.
	#REMOVES SMALL BLOBS
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, essential_kernel, iterations=1)   # Remove small noise

	return mask


def annotate_video_frame(
    frame: np.ndarray,
    contours: List[np.ndarray],
    *,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cx: float = 0.0,
    cy: float = 0.0,
    circle_radius_in: Optional[float] = None,
    label_3d: bool = False
) -> np.ndarray:
    """
    Annotate video frame with detected shapes, coordinates, and visual markers.
    
    Draws shape outlines, centroids, labels, and coordinate information on the frame.
    Provides both 2D pixel coordinates (relative to image center) and optional 3D
    world coordinates using the pinhole camera model.
    
    Args:
        frame: Input BGR frame to annotate
        contours: List of detected shape contours
        fx, fy: Camera focal lengths for 3D coordinate calculation
        cx, cy: Image center coordinates (coordinate system origin)
        circle_radius_in: Known real-world radius of reference circle
        label_3d: Whether to compute and display 3D world coordinates
        
    Returns:
        Annotated frame with shape outlines, labels, and coordinate information
    """
    annotated = frame.copy()

    # === COORDINATE SYSTEM ORIGIN ===
    # Draw green dot at image center to show coordinate system origin
    try:
        center_u = int(round(cx))
        center_v = int(round(cy))
        cv2.circle(annotated, (center_u, center_v), 6, (0, 255, 0), -1)  # Green dot
    except Exception:
        pass

    # === 3D COORDINATE SETUP ===
    # Retrieve cached depth and scale factors for 3D coordinate calculation
    Z_for_frame = None
    alpha_x = None
    alpha_y = None
    if label_3d and (fx is not None) and (fy is not None):
        global Z0_CACHE, ALPHA_X_CACHE, ALPHA_Y_CACHE
        if (Z0_CACHE is not None) and (ALPHA_X_CACHE is not None) and (ALPHA_Y_CACHE is not None):
            # Use pre-computed values from initialization
            Z_for_frame = Z0_CACHE
            alpha_x, alpha_y = ALPHA_X_CACHE, ALPHA_Y_CACHE
        elif circle_radius_in is not None:
            # Fallback: try to compute depth from current frame if not cached
            Z_try, _ = _estimate_depth_from_circle(contours, fx, fy, circle_radius_in)
            if Z_try is not None:
                Z0_CACHE = float(Z_try)
                ALPHA_X_CACHE = Z0_CACHE / float(fx)
                ALPHA_Y_CACHE = Z0_CACHE / float(fy)
                Z_for_frame = Z0_CACHE
                alpha_x, alpha_y = ALPHA_X_CACHE, ALPHA_Y_CACHE

    # === SHAPE ANNOTATION ===
    # Process each detected shape contour
    for idx, contour in enumerate(contours, start=1):
        shape_name = classify_shape(contour)

        # Draw shape outline with border and colored fill for visibility
        cv2.drawContours(annotated, [contour], -1, (0, 0, 0), thickness=4)      # Black border
        cv2.drawContours(annotated, [contour], -1, (0, 255, 255), thickness=2)  # Yellow outline

        # This section calculates the centroid (center of mass) of the contour using image moments.
		# cv2.moments(contour) computes spatial moments, which are essentially sums over the coordinates of the contour points.
		# In this context, since the contour is a set of points (not a grayscale image), the "pixel value" is just 1 for each point,
		# so the moments are not actually weighted by intensity—they are just sums over the coordinates.
		# "m10" is the sum of all x coordinates of the contour points, and "m01" is the sum of all y coordinates.
		# "m00" is the area (or, for a set of points, the number of points or the total "mass").
		# The centroid's x coordinate is m10/m00 (average x), and the y coordinate is m01/m00 (average y).
		# This is equivalent to taking the pure centroid of the shape's area, not just the average of the contour points.
		# Using moments gives a more accurate center for filled shapes, especially if the contour encloses an area.
        moments = cv2.moments(contour)
        if moments["m00"] != 0:  # Valid contour with non-zero area
            # Compute centroid coordinates
            u = int(moments["m10"] / moments["m00"])
            v = int(moments["m01"] / moments["m00"])
            
            # Draw centroid marker
            cv2.circle(annotated, (u, v), 8, (0, 0, 255), -1)  # Red dot
            cv2.drawMarker(annotated, (u, v), (255, 255, 255),  # White cross
                           markerType=cv2.MARKER_CROSS, markerSize=15, thickness=3)

            # === COORDINATE CALCULATION ===
            # Convert to center-origin coordinates (image center = (0,0))
            u_c = int(round(u - cx))
            v_c = int(round(v - cy))

            # Prepare text labels
            line1 = f"{shape_name} #{idx}  (u_c,v_c)=({u_c},{v_c})"  # 2D coordinates
            
            # Optional 3D world coordinates using pinhole camera model
            line2 = None
            if (Z_for_frame is not None) and (alpha_x is not None) and (alpha_y is not None):
                X = alpha_x * (u - cx)  # World X coordinate
                Y = alpha_y * (v - cy)  # World Y coordinate  
                Z = Z_for_frame         # Constant depth (planar assumption)
                line2 = f"(x,y,depth)=({X:.1f},{Y:.1f},{Z:.1f})"

            # === TEXT RENDERING ===
            # Draw text with black halo for readability on any background
            y0 = max(20, v - 20)  # Position text above centroid
            for (txt, dy) in [(line1, 0), (line2, 22)]:
                if txt is None:
                    continue
                pos = (u + 12, y0 + dy)
                # Black halo (thicker text)
                cv2.putText(annotated, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                # White text (thinner, on top)
                cv2.putText(annotated, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            # Fallback for invalid contours: use bounding box center
            x, y, _, _ = cv2.boundingRect(contour)
            label = f"{shape_name} #{idx}"
            cv2.putText(annotated, label, (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(annotated, label, (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated


def initialize_constant_depth_and_scales(
    first_frame: np.ndarray,
    background_lab: np.ndarray,
    mog2: cv2.BackgroundSubtractor,
    fx: float, fy: float, cx: float, cy: float,
    circle_radius_in: float,
    blur_kernel_size: int = 5,
    blur_sigma: float = 0.0,
    threshold: int = 95
) -> None:
    """
    Compute Z0 (constant depth) once using the circle in the first frame,
    then cache Z0 and the linear scales alpha_x = Z0/fx, alpha_y = Z0/fy.
    """
    global Z0_CACHE, ALPHA_X_CACHE, ALPHA_Y_CACHE

    # Build the same combined mask you use per-frame
    lab_mask = build_foreground_mask_with_background(first_frame, background_lab, blur_kernel_size, blur_sigma, threshold)

    fg = mog2.apply(first_frame)  # already warmed up by caller
    _, motion_mask = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)  # drop shadows
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, essential_kernel, iterations=1)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, essential_kernel, iterations=1)

    combined_mask = cv2.bitwise_or(lab_mask, motion_mask)
    contours = find_contours_with_min_area(combined_mask, min_area=10000, max_area=150000)

    # min area used to be 20000

    Z_try, _ = _estimate_depth_from_circle(contours, fx, fy, circle_radius_in)
    if Z_try is not None:
        Z0_CACHE = float(Z_try)
        ALPHA_X_CACHE = Z0_CACHE / float(fx)
        ALPHA_Y_CACHE = Z0_CACHE / float(fy)
        print(f"[Init] Z0={Z0_CACHE:.3f}, alpha_x={ALPHA_X_CACHE:.6f}, alpha_y={ALPHA_Y_CACHE:.6f}")
    else:
        print("[Init] Warning: could not estimate Z0 from the first frame (no circle found). "
              "3D labels will be omitted until depth is set.")



def process_video_stream(
    input_path: str, 
    output_path: str, 
    show_preview: bool = False,
    blur_kernel_size: int = 5,
    blur_sigma: float = 0.0,
    threshold: int = 95
) -> None:
    """
    Main video processing pipeline for shape detection and tracking.
    
    Processes an input video file frame by frame, applying the dual-detection
    algorithm (color + motion) to identify and track colored shapes on grass.
    Outputs an annotated video with shape labels and coordinate information.
    
    Processing Pipeline:
    1. Initialize background model from first frame
    2. Set up MOG2 background subtractor for motion detection  
    3. Calibrate 3D coordinate system using reference circle
    4. Process each frame with dual detection algorithm
    5. Annotate frames with shape information and coordinates
    6. Write annotated frames to output video
    
    Args:
        input_path: Path to input video file
        output_path: Path for output annotated video
        show_preview: Whether to display live preview window
        blur_kernel_size: Gaussian blur kernel size for noise reduction
        blur_sigma: Gaussian blur strength (0 = auto-calculate)
        threshold: LAB distance threshold for color-based detection
    """
    # === VIDEO SETUP ===
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
    print(f"Detection parameters: blur_kernel={blur_kernel_size}, blur_sigma={blur_sigma}, threshold={threshold}")

    # Set coordinate system origin at image center
    cx = width / 2.0
    cy = height / 2.0

    # Initialize output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # === BACKGROUND MODEL INITIALIZATION ===
    # Estimate grass background color from first frame border pixels
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame")
    background_lab = compute_background_lab(first_frame)
    print(f"Estimated grass background color (LAB): {background_lab}")

    # === MOTION DETECTION SETUP ===
    # Initialize MOG2 background subtractor for motion-based detection
    # history=100: considers last 100 frames for background model
    # varThreshold=25: sensitivity to pixel changes (lower = more sensitive)
    # detectShadows=True: identifies shadows as separate class (value 127)
    mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
    mog2.apply(first_frame)  # Initialize with first frame

    # === 3D CALIBRATION ===
    # Compute constant depth and scale factors using reference circle in first frame
    # This enables 3D world coordinate calculation for all subsequent frames
    initialize_constant_depth_and_scales(
        first_frame, background_lab, mog2,
        fx=DEFAULT_FX, fy=DEFAULT_FY, cx=cx, cy=cy,
        circle_radius_in=DEFAULT_CIRCLE_RADIUS_IN,
        blur_kernel_size=blur_kernel_size,
        blur_sigma=blur_sigma,
        threshold=threshold
    )

    # Reset video to beginning for main processing loop
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # === MAIN PROCESSING LOOP ===
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Apply dual detection algorithm to current frame
            annotated_frame, contours, combined_mask = process_video_frame(
                frame,
                background_lab,
                mog2,
                fx=DEFAULT_FX,
                fy=DEFAULT_FY,
                cx=cx,
                cy=cy,
                circle_radius_in=DEFAULT_CIRCLE_RADIUS_IN,
                label_3d=True,  # Enable 3D coordinate display
                blur_kernel_size=blur_kernel_size,
                blur_sigma=blur_sigma,
                threshold=threshold
            )

            # Write annotated frame to output video
            out.write(annotated_frame)

            # Optional live preview
            if show_preview:
                cv2.imshow('PennAir Shape Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break  # User pressed 'q' to quit

            # Progress reporting
            frame_count += 1
            if frame_count % 30 == 0:  # Report every 30 frames
                progress = frame_count / total_frames * 100
                print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
                
    finally:
        # Clean up resources
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()

    print(f"Video processing complete! Output saved to: {output_path}")
    print(f"Processed {frame_count} frames total")


def parse_video_args() -> argparse.Namespace:
	"""
	Parse command line arguments for video processing configuration.
	
	Provides tunable parameters for the detection algorithm, allowing users to
	optimize performance for different lighting conditions and object types.
	
	Returns:
		Parsed command line arguments with detection parameters
	"""
	parser = argparse.ArgumentParser(
		description="PennAir Challenge - Dynamic Shape Detection and Tracking",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  # Basic usage with default parameters
  python detect_shapes_video.py
  
  # Tune detection sensitivity
  python detect_shapes_video.py --threshold 70 --blur-kernel 9
  
  # Live preview mode
  python detect_shapes_video.py --preview
  
  # Process different video
  python detect_shapes_video.py --input "my_video.mp4" --output "result.mp4"

Parameter Tuning Guide:
  --threshold: Lower values = more sensitive color detection (50-100 typical range)
  --blur-kernel: Larger values = more noise reduction (5, 7, 9, 11 recommended)
  --blur-sigma: Higher values = stronger blur (0=auto, 1-6 typical range)
		"""
	)
	
	parser.add_argument("--input", "-i", type=str, 
					   default="PennAir 2024 App Dynamic Hard.mp4", 
					   help="Path to input video file")
	parser.add_argument("--output", "-o", type=str, 
					   default="dPennAir 2024 App Dynamic Hard Solution.mp4", 
					   help="Path to save annotated video")
	parser.add_argument("--preview", action="store_true", 
					   help="Show live preview while processing (press 'q' to quit)")
	parser.add_argument("--blur-kernel", type=int, default=7, 
					   help="Gaussian blur kernel size - larger reduces noise (must be odd)")
	parser.add_argument("--blur-sigma", type=float, default=4.0, 
					   help="Gaussian blur strength - higher = more blur (0 = auto-calculate)")
	parser.add_argument("--threshold", type=int, default=80, 
					   help="LAB distance threshold - lower = more sensitive detection (0-255)")
	return parser.parse_args()


# === PARAMETER HISTORY ===
# These are optimized parameter sets for different scenarios:
# Task 2 (original): blur-kernel=9, blur-sigma=6.0, threshold=95
# Task 4 (current):  blur-kernel=7, blur-sigma=4.0, threshold=80


if __name__ == "__main__":
	args = parse_video_args()
	
	# Validate input file exists
	if not os.path.exists(args.input):
		print(f"Error: Input video file not found: {args.input}")
		print("Please check the file path and try again.")
		exit(1)
	
	# Run the main processing pipeline
	process_video_stream(
		args.input, 
		args.output, 
		args.preview, 
		args.blur_kernel, 
		args.blur_sigma, 
		args.threshold
	)
