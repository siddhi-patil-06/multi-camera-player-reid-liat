import pandas as pd
import cv2
import os

def draw_tracks(frame, track_df, frame_id, id_mapping=None, is_broadcast=False):
    current = track_df[track_df['frame_id'] == frame_id]
    for _, row in current.iterrows():
        x1, y1, x2, y2 = int(row.x1), int(row.y1), int(row.x2), int(row.y2)
        tid = int(row.track_id)

        # Match ID from mapping
        if is_broadcast:
            label = f"BID {tid}"
        else:
            match = id_mapping.get(tid, None)
            label = f"TID {tid}" if match is None else f"TID {tid} ↔ BID {match}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    return frame

def create_combined_video(
    broadcast_video, tacticam_video,
    broadcast_tracks_csv, tacticam_tracks_csv,
    mapping_csv, output_path="output/matched_dual_view.mp4"
):
    cap_b = cv2.VideoCapture(broadcast_video)
    cap_t = cv2.VideoCapture(tacticam_video)

    df_b = pd.read_csv(broadcast_tracks_csv)
    df_t = pd.read_csv(tacticam_tracks_csv)
    mapping_df = pd.read_csv(mapping_csv)

    # Build mapping: tacticam_id → broadcast_id
    tac_to_bc = dict(zip(mapping_df.tacticam_track_id, mapping_df.broadcast_track_id))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap_b.get(cv2.CAP_PROP_FRAME_WIDTH)) + int(cap_t.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = max(int(cap_b.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap_t.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = 10
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_id = 0
    while cap_b.isOpened() and cap_t.isOpened():
        ret_b, frame_b = cap_b.read()
        ret_t, frame_t = cap_t.read()
        if not ret_b or not ret_t:
            break

        frame_b = draw_tracks(frame_b, df_b, frame_id, is_broadcast=True)
        frame_t = draw_tracks(frame_t, df_t, frame_id, tac_to_bc, is_broadcast=False)

        combined = cv2.hconcat([frame_b, frame_t])
        out.write(combined)
        frame_id += 1
        print(f"Processed frame {frame_id}", end="\r")

    cap_b.release()
    cap_t.release()
    out.release()
    print(f"\n🎉 Matched dual-view video saved to: {output_path}")

if __name__ == "__main__":
    create_combined_video(
        broadcast_video="data/broadcast.mp4",
        tacticam_video="data/tacticam.mp4",
        broadcast_tracks_csv="output/tracks_broadcast.csv",
        tacticam_tracks_csv="output/tracks_tacticam.csv",
        mapping_csv="output/player_mapping_final.csv"
    )
