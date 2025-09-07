import argparse
import os
from typing import List, Tuple

import cv2
import numpy as np


def read_image_bgr(image_path: str) -> np.ndarray:
	if not os.path.exists(image_path):
		raise FileNotFoundError(f"Input image not found: {image_path}")
	image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
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


def build_foreground_mask(image_bgr: np.ndarray) -> np.ndarray:
	"""Create a binary mask for solid shapes on grass using color distance in LAB."""
	blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)
	background_lab = compute_background_lab(blurred)

	lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)

	delta = np.linalg.norm(lab - background_lab.reshape(1, 1, 3), axis=2)
	delta_norm = cv2.normalize(delta, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

	_, mask = cv2.threshold(delta_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, essential_kernel, iterations=2)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, essential_kernel, iterations=1)

	return mask


def classify_shape(contour: np.ndarray) -> str:
	perimeter = float(cv2.arcLength(contour, True))
	if perimeter == 0:
		return "unknown"

	epsilon = 0.02 * perimeter
	approx = cv2.approxPolyDP(contour, epsilon, True)
	vertex_count = len(approx)

	area = float(cv2.contourArea(contour))
	if area <= 0:
		return "unknown"

	circularity = 4.0 * np.pi * area / (perimeter * perimeter + 1e-6)

	if vertex_count >= 8 or circularity > 0.78:
		return "circle"
	if vertex_count == 3:
		return "triangle"
	if vertex_count == 4:
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


def find_contours(mask: np.ndarray) -> List[np.ndarray]:
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	min_area = max(100, int(0.0005 * mask.shape[0] * mask.shape[1]))
	large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
	large_contours.sort(key=lambda c: cv2.boundingRect(c)[0])
	return large_contours


def annotate_image(image_bgr: np.ndarray, contours: List[np.ndarray]) -> np.ndarray:
	annotated = image_bgr.copy()
	for idx, contour in enumerate(contours, start=1):
		shape_name = classify_shape(contour)

		cv2.drawContours(annotated, [contour], -1, (0, 0, 0), thickness=3)
		cv2.drawContours(annotated, [contour], -1, (0, 255, 255), thickness=1)

		moments = cv2.moments(contour)
		if moments["m00"] != 0:
			center_x = int(moments["m10"] / moments["m00"])
			center_y = int(moments["m01"] / moments["m00"])
			cv2.circle(annotated, (center_x, center_y), 5, (0, 0, 255), -1)
			cv2.drawMarker(annotated, (center_x, center_y), (255, 255, 255), markerType=cv2.MARKER_CROSS, thickness=2)
			label = f"{shape_name} #{idx}"
			cv2.putText(annotated, label, (center_x + 8, center_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
			cv2.putText(annotated, label, (center_x + 8, center_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
		else:
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
	parser = argparse.ArgumentParser(description="Detect and annotate solid shapes on a grassy background.")
	parser.add_argument("--input", "-i", type=str, default="PennAir 2024 App Static.png", help="Path to input image")
	parser.add_argument("--output", "-o", type=str, default="annotated.png", help="Path to save annotated image")
	parser.add_argument("--mask", type=str, default="", help="Optional path to save the binary mask image")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	width, height, count = run(args.input, args.output, args.mask)
	print(f"Saved annotated image to: {args.output} ({width}x{height}), shapes detected: {count}")
