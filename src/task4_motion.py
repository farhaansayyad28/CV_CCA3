"""Task 4: consecutive-frame differencing, MOG2 (GMM), and Lucas-Kanade flow."""
import cv2
import numpy as np
import pandas as pd
from common import OUTPUT_DIR, save_image, find_video


def read_frame_at(cap, frame_number):
    """Read a specific zero-based frame from an open video capture."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"Could not read video frame {frame_number}.")
    return frame


def main():
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

    # Frame differencing: keep the original consecutive pair, frames 0 and 1.
    previous = read_frame_at(cap, 0)
    current = read_frame_at(cap, 1)
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

    difference = cv2.absdiff(previous_gray, current_gray)
    _, diff_mask = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
    save_image(out / "frame_0000.jpg", previous)
    save_image(out / "frame_0001.jpg", current)
    save_image(out / "frame_difference.jpg", difference)
    save_image(out / "frame_difference_mask.jpg", diff_mask)

    # MOG2: learn the background from the opening 20 frames.
    gmm = cv2.createBackgroundSubtractorMOG2(
        history=300, varThreshold=25, detectShadows=True
    )
    for index in range(min(20, count)):
        frame = read_frame_at(cap, index)
        mask = gmm.apply(frame)
    save_image(out / "gmm_foreground_mask.jpg", mask)
    foreground_mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
    save_image(
        out / "gmm_foreground.jpg",
        cv2.bitwise_and(frame, frame, mask=foreground_mask)
    )

    # Lucas-Kanade: use a clearer pair with visible motion, frames 300 and 315.
    # At about 59.94 FPS, these frames are approximately 5.00 s and 5.25 s.
    lk_start_index = 300
    lk_next_index = 315
    if count <= lk_next_index:
        raise ValueError(
            f"Video needs at least {lk_next_index + 1} frames for the selected "
            "Lucas-Kanade pair."
        )

    lk_previous = read_frame_at(cap, lk_start_index)
    lk_current = read_frame_at(cap, lk_next_index)
    lk_previous_gray = cv2.cvtColor(lk_previous, cv2.COLOR_BGR2GRAY)
    lk_current_gray = cv2.cvtColor(lk_current, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(
        lk_previous_gray,
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )

    flow_image = lk_current.copy()
    tracked = 0
    if corners is not None:
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            lk_previous_gray, lk_current_gray, corners, None
        )
        if next_points is not None and status is not None:
            for start, end, good in zip(
                corners.reshape(-1, 2),
                next_points.reshape(-1, 2),
                status.ravel()
            ):
                if good:
                    x1, y1 = np.round(start).astype(int)
                    x2, y2 = np.round(end).astype(int)
                    cv2.arrowedLine(
                        flow_image, (x1, y1), (x2, y2),
                        (0, 255, 0), 3, tipLength=0.35
                    )
                    cv2.circle(flow_image, (x2, y2), 7, (0, 0, 255), -1)
                    tracked += 1

    save_image(out / "lucas_kanade_vectors.jpg", flow_image)
    pd.DataFrame([{
        "Video_file": video.name,
        "FPS": round(fps, 3),
        "Frame_count": count,
        "Duration_seconds": round(count / fps, 3) if fps > 0 else "unknown",
        "Lucas_Kanade_start_frame": lk_start_index,
        "Lucas_Kanade_next_frame": lk_next_index,
        "Lucas_Kanade_start_time_seconds": round(lk_start_index / fps, 3) if fps > 0 else "unknown",
        "Lucas_Kanade_next_time_seconds": round(lk_next_index / fps, 3) if fps > 0 else "unknown",
        "Tracked_points": tracked
    }]).to_csv(out / "video_summary.csv", index=False)

    cap.release()
    print(f"Task 4 complete. Outputs: {out}")
    print(
        f"Lucas-Kanade used frames {lk_start_index} and {lk_next_index}; "
        f"tracked {tracked} points."
    )


if __name__ == "__main__":
    main()