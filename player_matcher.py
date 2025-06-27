import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

def load_features(path):
    df = pd.read_csv(path)
    df = df[df["duration"] >= 30]  # filter noisy/short tracks
    return df

def compute_similarity_and_match(tac_df, bc_df, output_csv, similarity_threshold=0.9):
    tac_ids = tac_df["track_id"].tolist()
    bc_ids = bc_df["track_id"].tolist()

    tac_features = tac_df[["avg_hist_R", "avg_hist_G", "avg_hist_B", "duration", "avg_width", "avg_height"]].values
    bc_features = bc_df[["avg_hist_R", "avg_hist_G", "avg_hist_B", "duration", "avg_width", "avg_height"]].values

    # Compute cosine similarity
    sim_matrix = cosine_similarity(tac_features, bc_features)

    # Store match candidates
    candidates = []
    for i, tac_id in enumerate(tac_ids):
        for j, bc_id in enumerate(bc_ids):
            score = sim_matrix[i][j]
            if score >= similarity_threshold:
                candidates.append({
                    "tacticam_track_id": tac_id,
                    "broadcast_track_id": bc_id,
                    "similarity_score": score,
                    "tacticam_duration": tac_df.iloc[i]["duration"]
                })

    # Sort by similarity score and duration
    df = pd.DataFrame(candidates)
    df = df.sort_values(by=["similarity_score", "tacticam_duration"], ascending=[False, False])

    # One-to-one matching: keep best unique match for each ID
    df = df.drop_duplicates(subset="broadcast_track_id", keep="first")
    df = df.drop_duplicates(subset="tacticam_track_id", keep="first")

    # Save
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df[["tacticam_track_id", "broadcast_track_id", "similarity_score"]].to_csv(output_csv, index=False)
    print(f"✅ Final refined mapping saved to: {output_csv}")

if __name__ == "__main__":
    tac_feat_path = "output/features_tacticam.csv"
    bc_feat_path = "output/features_broadcast.csv"
    output_path = "output/player_mapping_final.csv"

    tac_df = load_features(tac_feat_path)
    bc_df = load_features(bc_feat_path)

    compute_similarity_and_match(tac_df, bc_df, output_path, similarity_threshold=0.9)
