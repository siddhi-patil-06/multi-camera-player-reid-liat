import pandas as pd
import cv2
import os

def extract_first_crop(video_path, track_csv, track_id):
    df = pd.read_csv(track_csv)
    df = df[df['track_id'] == track_id]
    if df.empty:
        return None

    first_row = df.sort_values("frame_id").iloc[0]
    frame_id = int(first_row["frame_id"])
    x1, y1, x2, y2 = map(int, [first_row["x1"], first_row["y1"], first_row["x2"], first_row["y2"]])

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    crop = frame[y1:y2, x1:x2]
    return crop

def generate_comparison_images(mapping_csv, tacticam_csv, broadcast_csv, tacticam_video, broadcast_video, output_dir="output/matches_vis"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(mapping_csv)

    for _, row in df.iterrows():
        tac_id = int(row["tacticam_track_id"])
        bc_id = int(row["broadcast_track_id"])

        tac_crop = extract_first_crop(tacticam_video, tacticam_csv, tac_id)
        bc_crop = extract_first_crop(broadcast_video, broadcast_csv, bc_id)

        if tac_crop is None or bc_crop is None:
            print(f"⚠️ Skipping pair T{tac_id} - B{bc_id}: missing crop.")
            continue

        tac_crop = cv2.resize(tac_crop, (200, 300))
        bc_crop = cv2.resize(bc_crop, (200, 300))

        comparison = cv2.hconcat([tac_crop, bc_crop])
        filename = f"match_{tac_id}_{bc_id}.png"
        cv2.imwrite(os.path.join(output_dir, filename), comparison)
        print(f"✅ Saved: {filename}")

if __name__ == "__main__":
    generate_comparison_images(
        mapping_csv="output/player_mapping_final.csv",
        tacticam_csv="output/tracks_tacticam.csv",
        broadcast_csv="output/tracks_broadcast.csv",
        tacticam_video="data/tacticam.mp4",
        broadcast_video="data/broadcast.mp4"
    )
