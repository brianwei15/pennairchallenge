import argparse
import os
from typing import List, Tuple, Optional

import cv2
import numpy as np

# Import the functions from our static image detector
from detect_shapes import (
	compute_background_lab, 
	build_foreground_mask, 
	find_contours, 
	classify_shape, 
	essential_kernel
)

# --- Defaults for 3D labeling (you can override via CLI if you want) ---
DEFAULT_FX = 2564.3186869
DEFAULT_FY = 2569.70273111
DEFAULT_CIRCLE_RADIUS_IN = 10.0  # circle's real radius (inches)


# --- One-time calibration cache (computed once, reused forever) ---
Z0_CACHE: Optional[float] = None        # constant depth to camera
ALPHA_X_CACHE: Optional[float] = None   # Z0 / fx
ALPHA_Y_CACHE: Optional[float] = None   # Z0 / fy



def process_video_frame_mask_only(
    frame: np.ndarray,
    background_lab: Optional[np.ndarray] = None,
    blur_kernel_size: int = 5,
    blur_sigma: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Process a single video frame to show only the LAB mask for tuning."""
    # If no background provided, compute it from this frame
    if background_lab is None:
        print("No background provided, computing from frame")
        background_lab = compute_background_lab(frame)

    # Color-based mask with tunable blur parameters
    lab_mask = build_foreground_mask_with_background_tunable(
        frame, background_lab, blur_kernel_size, blur_sigma
    )

    # Convert single-channel mask to 3-channel for video output
    mask_rgb = cv2.cvtColor(lab_mask, cv2.COLOR_GRAY2BGR)

    return mask_rgb, lab_mask


def build_foreground_mask_with_background_tunable(
    image_bgr: np.ndarray, 
    background_lab: np.ndarray,
    blur_kernel_size: int = 5,
    blur_sigma: float = 0.0
) -> np.ndarray:
	"""Modified version with tunable blur parameters."""
	# Make sure kernel size is odd
	if blur_kernel_size % 2 == 0:
		blur_kernel_size += 1
	
	# Print the actual sigma value being used
	if blur_sigma == 0.0:
		# OpenCV auto-calculates sigma using: sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
		auto_sigma = 0.3 * ((blur_kernel_size - 1) * 0.5 - 1) + 0.8
	
	blurred = cv2.GaussianBlur(image_bgr, (blur_kernel_size, blur_kernel_size), blur_sigma)
	lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)

	# Use provided background instead of computing from border
	delta = np.linalg.norm(lab - background_lab.reshape(1, 1, 3), axis=2)
	delta_norm = cv2.normalize(delta, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

	threshold_value, mask = cv2.threshold(delta_norm, 110, 255, cv2.THRESH_BINARY)
	print(f"Otsu threshold: {threshold_value}")

    #95 for the threshold value for grass.

    #  + cv2.THRESH_OTSU

	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, essential_kernel, iterations=2)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, essential_kernel, iterations=1)

	return mask


def process_video_stream_mask_test(
    input_path: str, 
    output_path: str, 
    show_preview: bool = False,
    blur_kernel_size: int = 5,
    blur_sigma: float = 0.0
) -> None:
    """Process video showing only LAB mask for parameter tuning."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")
    print(f"Blur parameters: kernel_size={blur_kernel_size}, sigma={blur_sigma}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Estimate background from first frame for color model
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame")
    background_lab = compute_background_lab(first_frame)
    # Convert OpenCV LAB (0-255) to standard LAB ranges
    L_standard = background_lab[0] * 100.0 / 255.0  # 0-100
    A_standard = background_lab[1] - 128.0          # -128 to +127  
    B_standard = background_lab[2] - 128.0          # -128 to +127
    print(f"Estimated background color (OpenCV LAB): {background_lab}")
    print(f"Estimated background color (Standard LAB): L={L_standard:.1f}, A={A_standard:.1f}, B={B_standard:.1f}")

    # Reset to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            mask_frame, _ = process_video_frame_mask_only(
                frame,
                background_lab,
                blur_kernel_size,
                blur_sigma
            )

            out.write(mask_frame)

            if show_preview:
                cv2.imshow('LAB Mask', mask_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
    finally:
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()

    print(f"Mask video processing complete! Output saved to: {output_path}")
    print(f"Processed {frame_count} frames total")


def parse_video_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Test LAB mask with tunable Gaussian blur parameters.")
	parser.add_argument("--input", "-i", type=str, default="PennAir 2024 App Dynamic Hard.mp4", help="Path to input video file")
	parser.add_argument("--output", "-o", type=str, default="mask_test_output.mp4", help="Path to save mask video")
	parser.add_argument("--preview", action="store_true", help="Show live preview while processing (press 'q' to quit)")
	parser.add_argument("--blur-kernel", type=int, default=7, help="Gaussian blur kernel size (must be odd)")
	parser.add_argument("--blur-sigma", type=float, default=4.0, help="Gaussian blur sigma (0 means auto-calculate)")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_video_args()
	if not os.path.exists(args.input):
		print(f"Error: Input video file not found: {args.input}")
		print("Make sure the video file is in the current directory.")
		exit(1)
	
	process_video_stream_mask_test(
		args.input, 
		args.output, 
		args.preview,
		args.blur_kernel,
		args.blur_sigma
	)
