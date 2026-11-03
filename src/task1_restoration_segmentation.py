"""Task 1: noise, restoration metrics, thresholding, morphology, edges, watershed."""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from common import IMAGE_DIR, OUTPUT_DIR, read_image, require_file, save_image


def main():
    out = OUTPUT_DIR / "task1"
    out.mkdir(parents=True, exist_ok=True)
    # Load the original photo; keep BGR for OpenCV and make an RGB copy for scoring.
    bgr = read_image(require_file(IMAGE_DIR / "original.jpg", "the original photo"))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Gaussian noise: add small random changes across the image.
    rng = np.random.default_rng(42)
    noisy_gaussian = np.clip(rgb.astype(np.float32) + rng.normal(0, 20, rgb.shape), 0, 255).astype(np.uint8)

    # Salt-and-pepper noise: replace randomly selected pixels with black or white.
    noisy_sp = rgb.copy()
    count = int(0.04 * rgb.shape[0] * rgb.shape[1])
    ys, xs = rng.integers(0, rgb.shape[0], count), rng.integers(0, rgb.shape[1], count)
    noisy_sp[ys[:count//2], xs[:count//2]] = 255
    noisy_sp[ys[count//2:], xs[count//2:]] = 0

    # Apply Gaussian blur and median filtering to both noisy versions.
    gaussian_g = cv2.GaussianBlur(noisy_gaussian, (5, 5), 1.0)
    median_g = cv2.medianBlur(noisy_gaussian, 5)
    gaussian_sp = cv2.GaussianBlur(noisy_sp, (5, 5), 1.0)
    median_sp = cv2.medianBlur(noisy_sp, 5)
    # Compare the noisy and filtered images with the original using PSNR and SSIM.
    candidates = {
        "Gaussian noisy": noisy_gaussian, "Gaussian filtered": gaussian_g,
        "Median filtered (Gaussian noise)": median_g, "Salt-pepper noisy": noisy_sp,
        "Gaussian filtered (salt-pepper)": gaussian_sp, "Median filtered (salt-pepper)": median_sp,
    }
    rows = []
    for label, image in candidates.items():
        rows.append({"Image": label,
                     "PSNR_dB": round(peak_signal_noise_ratio(rgb, image, data_range=255), 3),
                     "SSIM": round(structural_similarity(rgb, image, channel_axis=2, data_range=255), 4)})
    # Save the measured scores in a CSV file.
    pd.DataFrame(rows).to_csv(out / "restoration_metrics.csv", index=False)

    # Save the original, noisy, and restored images for visual comparison.
    for name, image in {"original": rgb, "gaussian_noise": noisy_gaussian, "salt_pepper_noise": noisy_sp,
                        "gaussian_filtered": gaussian_g, "median_filtered_gaussian": median_g,
                        "gaussian_filtered_salt_pepper": gaussian_sp, "median_filtered_salt_pepper": median_sp}.items():
        save_image(out / f"{name}.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Convert the photo to grayscale and smooth it before thresholding.
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu chooses a threshold automatically to split pixels into two groups.
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphology: opening removes small specks; closing fills small gaps.
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    # Detect image edges with Canny.
    edges = cv2.Canny(blurred, 50, 150)

    # Watershed: make foreground/background markers to guide region separation.
    sure_bg = cv2.dilate(closing, kernel, iterations=3)
    dist = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    marked = cv2.watershed(bgr.copy(), markers)
    watershed = bgr.copy()
    watershed[marked == -1] = (0, 0, 255)
    # Save the grayscale, segmentation, edge, and watershed outputs.
    for name, img in {"gray": gray, "threshold_otsu": binary, "morphology_open_close": closing,
                      "canny_edges": edges, "watershed_boundaries": watershed}.items():
        save_image(out / f"{name}.png", img)
    print(f"Task 1 complete. Outputs: {out}")
    print("Use restoration_metrics.csv for your measured PSNR/SSIM results; discuss suitability using your own outputs.")


if __name__ == "__main__":
    main()
