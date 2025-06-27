import cv2
import pandas as pd
from deep_sort_realtime.deepsort_tracker import DeepSort
import os

def run_tracking(video_path, detections_csv, output_csv):
    df = pd.read_csv(detections_csv)
    df["frame_id"] = df["frame_id"].astype(int)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    tracker = DeepSort(
        max_age=30,         # how long to keep a lost track
        n_init=5,           # frames before confirming a track
        max_iou_distance=0.7,
        embedder="mobilenet",
        half=True,
        bgr=True,
        embedder_gpu=False,
    )

    output = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_dets = df[df["frame_id"] == frame_id]
        detections = []

        for _, row in frame_dets.iterrows():
            x1, y1, x2, y2 = int(row.x1), int(row.y1), int(row.x2), int(row.y2)
            conf = float(row.confidence)
            cls = row.class_name
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            output.append([frame_id, track_id, int(l), int(t), int(r), int(b)])

        frame_id += 1

    cap.release()
    out_df = pd.DataFrame(output, columns=["frame_id", "track_id", "x1", "y1", "x2", "y2"])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"✅ Tracking done: {output_csv}")


if __name__ == "__main__":
    run_tracking("data/broadcast.mp4", "output/detections_broadcast.csv", "output/tracks_broadcast.csv")
    run_tracking("data/tacticam.mp4", "output/detections_tacticam.csv", "output/tracks_tacticam.csv")
