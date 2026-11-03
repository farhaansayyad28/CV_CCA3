"""Task 2: HOG descriptors and SIFT matching across original and transformed photos."""
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
from common import IMAGE_DIR, OUTPUT_DIR, read_image, require_file, save_image


def to_gray_resized(image, size=(640, 480)):
    return cv2.cvtColor(cv2.resize(image, size), cv2.COLOR_BGR2GRAY)


def match_and_save(a, b, label, out):
    # Detect SIFT keypoints and describe the image areas around them.
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(a, None)
    kp2, des2 = sift.detectAndCompute(b, None)
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        raise ValueError(f"Not enough SIFT features for {label}. Try sharper, detailed photos of the same scene.")
    # Keep likely matches with the ratio test, then draw them for review.
    pairs = cv2.BFMatcher().knnMatch(des1, des2, k=2)
    good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
    drawing = cv2.drawMatches(a, kp1, b, kp2, good, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    save_image(out / f"{label}_sift_matches.jpg", drawing)
    return len(kp1), len(kp2), len(good)


def main():
    out = OUTPUT_DIR / "task2"
    out.mkdir(parents=True, exist_ok=True)
    # Load the two photos of the same object or scene.
    first = read_image(require_file(IMAGE_DIR / "original.jpg", "the original photo"))
    second = read_image(require_file(IMAGE_DIR / "second.jpg", "the second photo of the same object or scene"))
    # Use the same image size and grayscale format for both photos.
    gray1, gray2 = to_gray_resized(first), to_gray_resized(second)
    hog_rows = []
    # Extract HOG descriptors to summarize edge directions in each photo.
    for label, gray in (("original", gray1), ("second", gray2)):
        descriptor, visual = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                 visualize=True, block_norm="L2-Hys")
        hog_rows.append({"Image": label, "Descriptor_length": len(descriptor)})
        visual = exposure.rescale_intensity(visual, in_range=(0, 10))
        save_image(out / f"{label}_hog_visualization.png", np.uint8(255 * visual))
    # Create a rotated and resized version for the second SIFT comparison.
    center = (gray2.shape[1] / 2, gray2.shape[0] / 2)
    matrix = cv2.getRotationMatrix2D(center, 25, 0.8)
    transformed = cv2.warpAffine(gray2, matrix, (gray2.shape[1], gray2.shape[0]))
    save_image(out / "second_rotated_resized.png", transformed)
    stats = []
    # Match features in the original pair and the transformed pair.
    for label, other in (("original_pair", gray2), ("rotated_resized_pair", transformed)):
        k1, k2, good = match_and_save(gray1, other, label, out)
        stats.append({"Comparison": label, "Keypoints_photo1": k1, "Keypoints_photo2": k2, "Good_matches": good})
    # Save HOG descriptor lengths and SIFT matching counts.
    pd.DataFrame(hog_rows).to_csv(out / "hog_summary.csv", index=False)
    pd.DataFrame(stats).to_csv(out / "sift_matching_summary.csv", index=False)
    print(f"Task 2 complete. Outputs: {out}")
    print("Record your actual keypoint and match counts from the CSV files in the report.")


if __name__ == "__main__":
    main()
