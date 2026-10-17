"""EN3160 A2 Q1: detect sunflower blobs using scale-normalized LoG extrema.

Usage:
  python src/q1_blob_detection.py
  python src/q1_blob_detection.py --sigma-min 2 --sigma-max 16 --sigma-count 22
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_laplace, maximum_filter


def detect_blobs(gray: np.ndarray, sigmas: np.ndarray, threshold: float):
    """Return (x, y, sigma, response) maxima in (scale, y, x) space."""
    image = gray.astype(np.float32) / 255.0
    # Negative LoG gives positive peaks for light spots on dark background.
    responses = np.stack(
        [-(s * s) * gaussian_laplace(image, sigma=float(s)) for s in sigmas]
    )
    peaks = responses == maximum_filter(responses, size=(3, 11, 11))
    peaks &= responses > threshold
    scale_idx, ys, xs = np.nonzero(peaks)
    return np.column_stack([xs, ys, sigmas[scale_idx], responses[scale_idx, ys, xs]])


def suppress_overlaps(blobs: np.ndarray, overlap: float = 0.5) -> np.ndarray:
    """Greedy spatial suppression; keep stronger responses first."""
    if len(blobs) == 0:
        return blobs
    ranked = blobs[np.argsort(-blobs[:, 3])]
    selected = []
    for candidate in ranked:
        x, y, sigma, _ = candidate
        radius = np.sqrt(2.0) * sigma
        if all(
            np.hypot(x - prev[0], y - prev[1]) >= overlap * (radius + np.sqrt(2.0) * prev[2])
            for prev in selected
        ):
            selected.append(candidate)
    return np.asarray(selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="images/the_berry_farms_sunflower_field.jpeg")
    parser.add_argument("--sigma-min", type=float, default=2.0)
    parser.add_argument("--sigma-max", type=float, default=16.0)
    parser.add_argument("--sigma-count", type=int, default=22)
    parser.add_argument("--threshold", type=float, default=0.025)
    parser.add_argument("--largest", type=int, default=10)
    args = parser.parse_args()
    if args.sigma_min <= 0 or args.sigma_max <= args.sigma_min or args.sigma_count < 3:
        parser.error("Choose 0 < sigma-min < sigma-max and sigma-count >= 3")

    path = Path(args.image)
    image = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}. Extract the assignment images first.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sigmas = np.linspace(args.sigma_min, args.sigma_max, args.sigma_count)
    blobs = suppress_overlaps(detect_blobs(gray, sigmas, args.threshold))
    blobs = blobs[np.argsort(-blobs[:, 2])] if len(blobs) else blobs

    out = Path("results/q1")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "circles.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["center_x", "center_y", "sigma", "radius_px", "log_response"])
        for x, y, sigma, strength in blobs:
            writer.writerow([round(x, 2), round(y, 2), round(sigma, 3),
                             round(np.sqrt(2) * sigma, 3), round(strength, 6)])

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for x, y, sigma, strength in blobs:
        ax.add_patch(plt.Circle((x, y), np.sqrt(2) * sigma,
                                fill=False, edgecolor="lime", linewidth=1))
    ax.set_title(f"Scale-normalized LoG: {len(blobs)} blobs, sigma {sigmas[0]:g}–{sigmas[-1]:g}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "detected_circles.png", dpi=200)
    plt.close(fig)
    print(f"Detected {len(blobs)} blobs. Largest {args.largest} by radius:")
    for x, y, sigma, strength in blobs[:args.largest]:
        print(f"center=({x:.0f}, {y:.0f}), sigma={sigma:.2f}, "
              f"radius={np.sqrt(2)*sigma:.2f}px, response={strength:.4f}")
    print(f"Saved {out / 'detected_circles.png'} and {out / 'circles.csv'}")


if __name__ == "__main__":
    main()
