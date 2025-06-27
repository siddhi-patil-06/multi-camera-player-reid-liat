# detector.py

from ultralytics import YOLO
import cv2
import os
import pandas as pd

def detect_players(video_path, model, output_csv):
    cap = cv2.VideoCapture(video_path)

    detections = []
    frame_id = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)[0]  # run detection on the frame

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls_id = result
            class_name = model.names[int(cls_id)]

            # Only keep 'player' or 'ball' class
            if class_name in ["player", "ball"]:
                detections.append([
                    frame_id, int(x1), int(y1), int(x2), int(y2), round(conf, 3), class_name
                ])

        frame_id += 1

    cap.release()

    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Save to CSV
    df = pd.DataFrame(detections, columns=["frame_id", "x1", "y1", "x2", "y2", "confidence", "class_name"])
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved: {output_csv}")


if __name__ == "__main__":
    # 📦 CONFIGURATION
    yolo_model_path = "data/yolov11_weights.pt"
    broadcast_video_path = "data/broadcast.mp4"
    tacticam_video_path = "data/tacticam.mp4"

    # ⏬ LOAD MODEL
    print("🔍 Loading YOLOv11 model...")
    model = YOLO(yolo_model_path)

    # 🧠 RUN DETECTION
    print("🚀 Detecting in broadcast video...")
    detect_players(broadcast_video_path, model, "output/detections_broadcast.csv")

    print("🚀 Detecting in tacticam video...")
    detect_players(tacticam_video_path, model, "output/detections_tacticam.csv")

    print("✅ Detection complete.")
