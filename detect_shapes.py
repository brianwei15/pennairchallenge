import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np


def read_image_bgr(image_path: str) -> np.ndarray:
	if not os.path.exists(image_path):
		raise FileNotFoundError(f"Input image not found: {image_path}")
	image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR) #cv2.imread loads the image, and cv2.IMREAD_COLOR loads the image in full color
	if image_bgr is None:
		raise ValueError(f"Failed to load image: {image_path}")
	return image_bgr


def compute_background_lab(image_bgr: np.ndarray, border: int =12) -> np.ndarray:
	"""Estimate grassy background color in CIE LAB by sampling the border."""
	image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
	h, w = image_lab.shape[:2]

	mask = np.zeros((h, w), dtype=np.uint8)
	mask[:border, :] = 1
	mask[-border:, :] = 1
	mask[:, :border] = 1
	mask[:, -border:] = 1

	bg_pixels = image_lab[mask.astype(bool)]
	if bg_pixels.size == 0:
		bg_pixels = image_lab.reshape(-1, 3)
	background_lab = np.median(bg_pixels, axis=0).astype(np.float32)
	return background_lab


essential_kernel = np.ones((5, 5), dtype=np.uint8)

#takes a color image in an nd array and returns a binary mask in an nd array
def build_foreground_mask(image_bgr: np.ndarray) -> np.ndarray:
	"""Create a binary mask for solid shapes on grass using color distance in LAB."""
	blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0) #applies a matrix of size 5x5 with a Gaussian distribution, and auto calculated sigma (standard deviation)
	background_lab = compute_background_lab(blurred)

	#converts the blurred image to the LAB color space
	lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)

	#calculates the distance between each pixel in the blurred image and the background color in the LAB color space
	delta = np.linalg.norm(lab - background_lab.reshape(1, 1, 3), axis=2)
	#normalizes the distance between 0 and 255
	delta_norm = cv2.normalize(delta, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

	#applies a threshold to the normalized distance to create a binary mask
	#A mask is a binary image that where pictures where the color difference is greater 
	# than the threshold are white, and pictures where the color difference is less than the threshold are black
	_, mask = cv2.threshold(delta_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	#applies a "dilation", then "erosion" to the mask to fill small holes inside detected shapes and joins nearby white regions
	#FILLS HOLES
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, essential_kernel, iterations=2)

	#applies a "erosion", then "dilation" to the mask to remove small blobs left in the grass.
	#REMOVES SMALL BLOBS
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, essential_kernel, iterations=1)

	return mask

#classifies the shape given the contour
def classify_shape(contour: np.ndarray) -> str:
	# Calculates the perimeter (arc length) of the contour in pixel units.
	perimeter = float(cv2.arcLength(contour, True))
	if perimeter == 0:
		return "unknown"
		
	# epsilon determines the maximum distance between the original contour and its approximation.
	# A smaller epsilon yields a closer fit to the original contour, while a larger epsilon results in a simpler, more polygonal shape.
	epsilon = 0.02 * perimeter
	# approx is an array containing the coordinates of the corner points (vertices) of the polygonal approximation of the contour.
	approx = cv2.approxPolyDP(contour, epsilon, True)
	vertex_count = len(approx)

	area = float(cv2.contourArea(contour))
	if area <= 0:
		return "unknown"

	# Calculates the circularity of the contour.
	# Circularities > 0.78 are considered circles.
	circularity = 4.0 * np.pi * area / (perimeter * perimeter + 1e-6)

	if vertex_count >= 8 or circularity > 0.78:
		return "circle"
	if vertex_count == 3:
		return "triangle"
	if vertex_count == 4:
		
		# This section computes the bounding rectangle for the approximated polygon (approx).
		# The function cv2.boundingRect returns the x and y coordinates of the top-left corner,
		# as well as the width (w) and height (h) of the rectangle that tightly encloses the polygon.
		x, y, w, h = cv2.boundingRect(approx)
		aspect_ratio = w / float(h)
		if 0.90 <= aspect_ratio <= 1.10:
			return "square"
		return "quadrilateral"
	if vertex_count == 5:
		return "pentagon"
	if vertex_count == 6:
		return "hexagon"
	return f"{vertex_count}-gon"


#finds the contours of the shapes in the mask
def find_contours(mask: np.ndarray) -> List[np.ndarray]:
	#finds the contours of the shapes in the mask that are external (ignores holes and nested contours). 
	#cv2.RETR_EXTERNAL is used to find the external contours only.
	#cv2.CHAIN_APPROX_SIMPLE is used to only return the corners of the contours, saving memory, increasing efficiency
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#filters out contours that are too small
	min_area = max(100, int(0.0005 * mask.shape[0] * mask.shape[1]))
	large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

	#sorts the contours by the x-coordinate of the bounding rectangle of the contour
	#cv2.boundingRect(c)[0] extracts the x-coordinate of the bounding rectangle of each contour
	#which is used to sort them left to right.
	large_contours.sort(key=lambda c: cv2.boundingRect(c)[0])
	return large_contours


def annotate_image(image_bgr: np.ndarray, contours: List[np.ndarray]) -> np.ndarray:
	annotated = image_bgr.copy()
	for idx, contour in enumerate(contours, start=1):
		#classifies the shape
		shape_name = classify_shape(contour)

		#draws 2 contours
		cv2.drawContours(annotated, [contour], -1, (0, 0, 0), thickness=3)
		cv2.drawContours(annotated, [contour], -1, (0, 255, 255), thickness=1)


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
		if moments["m00"] != 0:
			center_x = int(moments["m10"] / moments["m00"])
			center_y = int(moments["m01"] / moments["m00"])

			#draws a circle and a cross at the center of the contour
			cv2.circle(annotated, (center_x, center_y), 5, (0, 0, 255), -1)
			cv2.drawMarker(annotated, (center_x, center_y), (255, 255, 255), markerType=cv2.MARKER_CROSS, thickness=2)
			label = f"{shape_name} #{idx}"
			#writes label twice to make it more readable
			cv2.putText(annotated, label, (center_x + 8, center_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
			cv2.putText(annotated, label, (center_x + 8, center_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
		else: #fallback for extremely thin contours with area approx 0.
			x, y, _, _ = cv2.boundingRect(contour)
			label = f"{shape_name} #{idx}"
			cv2.putText(annotated, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
			cv2.putText(annotated, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

	return annotated


def run(input_path: str, output_path: str, save_mask_path: str = "") -> Tuple[int, int, int]:
	image_bgr = read_image_bgr(input_path) 
	mask = build_foreground_mask(image_bgr)
	contours = find_contours(mask)
	annotated = annotate_image(image_bgr, contours)

	os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
	cv2.imwrite(output_path, annotated)
	if save_mask_path:
		os.makedirs(os.path.dirname(save_mask_path) or ".", exist_ok=True)
		cv2.imwrite(save_mask_path, mask)

	return annotated.shape[1], annotated.shape[0], len(contours)


def parse_args() -> argparse.Namespace:
	# 'parser' is an instance of argparse.ArgumentParser.
	# We use it here to define and handle command-line arguments for the script.
	# This allows users to specify input/output file paths and options when running the script from the command line.
	parser = argparse.ArgumentParser(description="Detect and annotate solid shapes on a grassy background.")
	parser.add_argument("--input", "-i", type=str, default="PennAir 2024 App Static.png", help="Path to input image")
	parser.add_argument("--output", "-o", type=str, default="annotated.png", help="Path to save annotated image")
	parser.add_argument("--mask", type=str, default="", help="Optional path to save the binary mask image")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	width, height, count = run(args.input, args.output, args.mask)
	print(f"Saved annotated image to: {args.output} ({width}x{height}), shapes detected: {count}")
