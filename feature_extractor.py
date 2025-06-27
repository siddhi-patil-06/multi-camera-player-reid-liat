import cv2
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def extract_track_features(video_path, tracks_csv, output_csv):
    df = pd.read_csv(tracks_csv)
    df["frame_id"] = df["frame_id"].astype(int)

    cap = cv2.VideoCapture(video_path)
    frame_cache = {}

    # Collect per-track stats
    track_data = defaultdict(list)

    print(f"🔍 Processing {os.path.basename(video_path)}...")
    for frame_id in tqdm(sorted(df["frame_id"].unique())):
        if frame_id in frame_cache:
            frame = frame_cache[frame_id]
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_cache[frame_id] = frame

        for _, row in df[df["frame_id"] == frame_id].iterrows():
            tid = row.track_id
            x1, y1, x2, y2 = map(int, [row.x1, row.y1, row.x2, row.y2])
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            avg_color = cv2.mean(crop)[:3]  # (B, G, R)
            h, w = crop.shape[:2]
            track_data[tid].append({
                "R": avg_color[2],
                "G": avg_color[1],
                "B": avg_color[0],
                "width": w,
                "height": h
            })

    cap.release()

    rows = []
    for tid, samples in track_data.items():
        arr = pd.DataFrame(samples)
        rows.append([
            tid,
            arr["R"].mean(),
            arr["G"].mean(),
            arr["B"].mean(),
            len(samples),                      # duration
            arr["width"].mean(),
            arr["height"].mean()
        ])

    out_df = pd.DataFrame(rows, columns=[
        "track_id", "avg_hist_R", "avg_hist_G", "avg_hist_B",
        "duration", "avg_width", "avg_height"
    ])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"✅ Saved: {output_csv}")


if __name__ == "__main__":
    extract_track_features(
        "data/broadcast.mp4",
        "output/tracks_broadcast.csv",
        "output/features_broadcast.csv"
    )
    extract_track_features(
        "data/tacticam.mp4",
        "output/tracks_tacticam.csv",
        "output/features_tacticam.csv"
    )
