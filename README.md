# EN3160 Assignment 2 — Fitting and Alignment

Python/OpenCV implementations for EN3160 Assignment 2 (October 2026).

## Tasks
1. Multiscale Laplacian-of-Gaussian blob detection on a sunflower image (no HoughCircles).
2. Custom line and circle RANSAC with fitting refinements.
3. Planar homography and overlay compositing.
4. SIFT feature matching, alignment, and differences for PCB images.

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Extract the instructor-provided archive to `images/`; put PCB images into `images/pcbs/`.
Use `python src/q1_blob_detection.py` for the first experiment.
Figures and tabular parameters are written to `results/q1/`.

## Reproducibility
Record parameters and actual measured outputs in the final report. See each script for documented defaults.
