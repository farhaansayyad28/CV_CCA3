# Computer Vision CCA 3

Beginner-friendly Python project for image restoration and segmentation, HOG/SIFT features, Haar/YOLO detection, and video motion estimation.

## 1. Add your own input media

Put these three files in the indicated locations. Keep the filenames exactly as shown (or change the constants near the top of the relevant script).

| File | Location | What it must contain |
|---|---|---|
| `original.jpg` | `data/input/images/` | Your original, unfiltered photo. Used for Task 1 and as the first Task 2 image. |
| `second.jpg` | `data/input/images/` | A second photo of the **same object or scene**. A different angle is useful, but not essential. Task 2 also creates its own rotated/resized variant. |
| `motion_video.mp4` or `motion_video.mov` | `data/input/video/` | Your own 15–30 second clip with a moving person/object and some stationary background. Both formats are accepted. |

Use a sharp, well-lit photo with visible texture and edges. Avoid using another person's reference data. The video can show something entirely different from the photos. No sample media or fabricated results are included.

## 2. Set up in Windows / VS Code

1. Open this `Computer_Vision_CCA3` folder in VS Code.
2. Open **Terminal → New Terminal**.
3. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt in VS Code and run `.venv\Scripts\activate.bat`.

4. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Run one task at a time from the project folder:

   ```powershell
   python src\task1_restoration_segmentation.py
   python src\task2_hog_sift.py
   python src\task3_detection.py
   python src\task4_motion.py
   ```

Task 3 uses the small pretrained `yolov8n.pt` model. Ultralytics downloads the model weights the first time that script runs, so an internet connection is needed for that initial download. The Haar frontal-face cascade is supplied by OpenCV. Haar detects faces; YOLO can detect several general object categories. If Haar finds no faces or YOLO finds no objects, report that result honestly.

## 3. What each task saves

All generated files go under `outputs/`. The scripts do not modify your input media.

- **Task 1:** seeded Gaussian and salt-and-pepper noise; Gaussian and median restoration; PSNR/SSIM CSV; Otsu threshold; opening/closing morphology; Canny edges; watershed boundary visualization.
- **Task 2:** HOG visualizations and descriptor lengths; SIFT match images for the supplied photo pair and for a rotated/resized version; keypoint and good-match counts in CSV files.
- **Task 3:** Haar face boxes, YOLO photo boxes and detection CSV. If your video is present, YOLO also analyzes one middle frame.
- **Task 4:** sample consecutive frames, frame difference/mask, MOG2/GMM foreground mask, Lucas–Kanade vectors, and video metadata CSV.

The numbers and images depend on your own media and machine. Outputs are not prefilled or represented as assignment results. Review each generated image and CSV and enter your actual observations in `REPORT_TEMPLATE.md`.

## Project layout

```text
Computer_Vision_CCA3/
├── data/input/images/       # add original.jpg and second.jpg
├── data/input/video/        # add motion_video.mp4
├── outputs/                 # generated results (starts empty)
├── src/                     # four runnable task scripts + shared helpers
├── requirements.txt
├── README.md
└── REPORT_TEMPLATE.md
```

## Troubleshooting

- **File not found:** confirm the exact filename, extension, and folder above. Windows may hide extensions; avoid names such as `original.jpg.jpg`.
- **Video cannot open:** try an MP4 encoded with H.264 and verify it plays in a media player.
- **Not enough SIFT features:** use clear, detailed photos with textured regions, and avoid blur or large occlusions.
- **YOLO download/import issue:** confirm `pip install -r requirements.txt` completed; allow the first model download to finish.
- **Results are poor:** try more even lighting, a visible stationary background, and a stable camera. Record what happened rather than altering results to match expectations.

See `REPORT_TEMPLATE.md` for the report structure and the fields that must be completed with your own details.
