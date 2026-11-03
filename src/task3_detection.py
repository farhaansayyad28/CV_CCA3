"""Task 3: Viola-Jones Haar face cascade and pretrained YOLO object detection."""
import cv2
import pandas as pd
from ultralytics import YOLO
from common import IMAGE_DIR, OUTPUT_DIR, read_image, require_file, save_image, find_video


def main():
    out = OUTPUT_DIR / "task3"
    out.mkdir(parents=True, exist_ok=True)
    # Load the photo used by both detectors.
    image = read_image(require_file(IMAGE_DIR / "original.jpg", "the original photo"))

    # Run the Haar cascade, which is trained to find frontal faces.
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError("OpenCV could not load its bundled frontal-face cascade.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    haar = image.copy()
    for x, y, w, h in faces:
        cv2.rectangle(haar, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Save the face boxes and their coordinates for review.
    save_image(out / "haar_faces.jpg", haar)
    pd.DataFrame([{"Detection": i + 1, "x": int(x), "y": int(y), "width": int(w), "height": int(h)}
                  for i, (x, y, w, h) in enumerate(faces)]).to_csv(out / "haar_detections.csv", index=False)

    # Run pretrained YOLO on the photo and keep its predicted boxes and labels.
    model = YOLO("yolov8n.pt")
    result = model.predict(source=str(IMAGE_DIR / "original.jpg"), conf=0.25, verbose=False)[0]
    save_image(out / "yolo_photo.jpg", result.plot())
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            cls = int(box.cls.item())
            detections.append({"Class": model.names[cls], "Confidence": round(float(box.conf.item()), 4),
                               "x1": round(float(box.xyxy[0][0]), 1), "y1": round(float(box.xyxy[0][1]), 1),
                               "x2": round(float(box.xyxy[0][2]), 1), "y2": round(float(box.xyxy[0][3]), 1)})
    pd.DataFrame(detections, columns=["Class", "Confidence", "x1", "y1", "x2", "y2"]).to_csv(out / "yolo_detections.csv", index=False)
    # If a video is available, run YOLO on a representative middle frame.
    video_path = find_video()
    if video_path is not None:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(total // 2, 0))
            ok, frame = cap.read()
            cap.release()
            if ok:
                frame_result = model.predict(source=frame, conf=0.25, verbose=False)[0]
                save_image(out / "yolo_video_frame.jpg", frame_result.plot())
    print(f"Task 3 complete. Haar faces found: {len(faces)}. YOLO outputs: {out}")
    print("A zero-detection result is valid; record what you actually observe.")


if __name__ == "__main__":
    main()
