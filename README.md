# 🏃‍♂️ Multi-Camera Player Re-Identification (Liat.ai Assignment)

This project uses a player re-identification pipeline for sports videos from two camera views (broadcast and tacticam). The goal is to assign consistent player IDs across both views using detection, tracking, feature extraction, and visual matching.

---

## 📁 Project Structure

```
player_reid_system/
├── data/
│   ├── broadcast.mp4
│   ├── tacticam.mp4
│   └── yolov11_weights.pt
├── output/
│   ├── detections_broadcast.csv
│   ├── detections_tacticam.csv
│   ├── tracks_broadcast.csv
│   ├── tracks_tacticam.csv
│   ├── features_broadcast.csv
│   ├── features_tacticam.csv
│   ├── player_mapping_final.csv
│   ├── matched_dual_view.mp4
│   ├── player_matches_video.mp4
│   └── matches_vis/
├── detector.py
├── tracker.py
├── feature_extractor.py
├── player_matcher.py
├── match_video_generator.py
├── annotate_videos.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository and create a virtual environment:

```bash
git clone https://github.com/siddhi-patil-06/multi-camera-player-reid-liat.git
cd player_reid_system
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows
```

### 2. Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Add the required files into the `data/` folder:

- `broadcast.mp4`
- `tacticam.mp4`
- `yolov11_weights.pt`

---

## 🚀 How to Run the Pipeline

```bash
# Step 1: Run detection using YOLOv11
python detector.py

# Step 2: Track players using DeepSORT
python tracker.py

# Step 3: Extract RGB/size features per player
python feature_extractor.py

# Step 4: Match player IDs across views using cosine similarity
python player_matcher.py

# Step 5: Generate dual-view annotated output video
python match_video_generator.py
```

---

## 🧠 Methodology

- **Detection**: YOLOv11 trained to detect players and the ball.
- **Tracking**: DeepSORT (with MobileNet) assigns consistent track IDs per player.
- **Feature Extraction**: Mean color (RGB), width, height, and track duration.
- **Matching**: Cosine similarity with duration filtering and one-to-one greedy matching.
- **Visualization**: Combined output with `BID` and `TID` visual mapping.

---

## 📦 Dependencies

Installed via `requirements.txt`. Includes:

- `ultralytics`
- `deep-sort-realtime`
- `opencv-python`
- `torch`, `torchvision`
- `numpy`, `pandas`, `scikit-learn`
- `matplotlib`, `tqdm`

---

## 📤 Outputs

| Output File | Description |
|-------------|-------------|
| `detections_*.csv` | YOLOv11 detections per frame |
| `tracks_*.csv` | DeepSORT-tracked player IDs |
| `features_*.csv` | Extracted per-player feature vectors |
| `player_mapping_final.csv` | Final matched IDs between views |
| `matched_dual_view.mp4` | Combined view showing tracked players |
| `matches_vis/` | Cropped comparisons of matched players |

---

## 📝 Notes

- Tracks with fewer than **30 frames** are excluded as noise.
- Matching ensures unique best-match IDs per view.
- You can modify the similarity threshold in `player_matcher.py`.

---

## 👤 Author

**Siddhi Patil**  
📧 siddhipatil064@gmail.com  
🔗 GitHub: [@siddhi-patil-06](https://github.com/siddhi-patil-06)

---

## ✅ Final Thoughts

This solution is designed to be modular, interpretable, and adaptable to real-world multi-view sports footage. Despite camera variation and occlusion, the pipeline achieves consistent cross-camera player re-identification through interpretable and scalable methods.
