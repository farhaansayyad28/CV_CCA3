"""Task 4: consecutive-frame differencing, MOG2 (GMM), and Lucas-Kanade flow."""
import cv2
import numpy as np
import pandas as pd
from common import OUTPUT_DIR, save_image, find_video


def main():
    # Find and open the supplied video.
    video = find_video()
    if video is None:
        raise FileNotFoundError(
            f"No video found in {OUTPUT_DIR.parent / 'data' / 'input' / 'video'}. "
            "Add motion_video.mp4 or motion_video.mov and run again."
        )
    out = OUTPUT_DIR / "task4"
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Read the first two frames for differencing and optical flow.
    ok, previous = cap.read()
    if not ok:
        cap.release()
        raise ValueError("The video has no readable frames.")
    # Set up MOG2 and let it learn the background from the opening frames.
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    gmm = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)
    # Give the background model a few frames to learn the scene.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(min(20, count)):
        ok, frame = cap.read()
        if not ok: break
        gmm.apply(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    ok, current = cap.read()
    if not ok:
        cap.release()
        raise ValueError("At least two video frames are required.")
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    # Compare consecutive grayscale frames and threshold their changes.
    difference = cv2.absdiff(previous_gray, current_gray)
    _, diff_mask = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
    save_image(out / "frame_0000.jpg", previous)
    save_image(out / "frame_0001.jpg", current)
    save_image(out / "frame_difference.jpg", difference)
    save_image(out / "frame_difference_mask.jpg", diff_mask)

    # Apply a fresh background model to initial frames, then save its mask.
    cap.release()
    cap = cv2.VideoCapture(str(video))
    gmm = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)
    for index in range(min(20, count)):
        ok, frame = cap.read()
        if not ok: break
        mask = gmm.apply(frame)
    save_image(out / "gmm_foreground_mask.jpg", mask)
    save_image(out / "gmm_foreground.jpg", cv2.bitwise_and(frame, frame, mask=cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]))

    # Select trackable corners and follow their movement into the next frame.
    corners = cv2.goodFeaturesToTrack(previous_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    flow_image = current.copy()
    tracked = 0
    if corners is not None:
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, corners, None)
        if next_points is not None and status is not None:
            for start, end, good in zip(corners.reshape(-1, 2), next_points.reshape(-1, 2), status.ravel()):
                if good:
                    x1, y1 = np.round(start).astype(int); x2, y2 = np.round(end).astype(int)
                    cv2.arrowedLine(flow_image, (x1, y1), (x2, y2), (0, 255, 0), 2, tipLength=0.3)
                    cv2.circle(flow_image, (x2, y2), 3, (0, 0, 255), -1)
                    tracked += 1
    save_image(out / "lucas_kanade_vectors.jpg", flow_image)
    pd.DataFrame([{"Video_file": video.name, "FPS": round(fps, 3), "Frame_count": count,
                   "Duration_seconds": round(count / fps, 3) if fps > 0 else "unknown",
                   "Tracked_points_first_pair": tracked}]).to_csv(out / "video_summary.csv", index=False)
    cap.release()
    print(f"Task 4 complete. Outputs: {out}")
    print("The saved masks/vectors are sample frames; inspect the video and record your observations.")


if __name__ == "__main__":
    main()
