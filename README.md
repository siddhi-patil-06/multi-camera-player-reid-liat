# multi-camera-player-reid-liat
# ♂️ Player Re-Identification System

The project uses a player re-identification pipeline for sports videos from two camera views (broadcast and tacticam). The purpose is to give stable player IDs between both views using detection, tracking, feature extraction, and visual matching.

---

##  Project Structure

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

---

## ⚙️ Setup Instructions

### 1. Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/your-username/player_reid_system.git
cd player_reid_system
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows
```

### 2. Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Put necessary files in the `data/` directory:

- `broadcast.mp4`
- `tacticam.mp4`
- `yolov11_weights.pt` (fine-tuned YOLOv11 weights)

---

##  How to Run

Execute each pipeline step sequentially:

```bash
# Step 1: Object Detection
python detector.py

# Step 2: DeepSORT Tracking
python tracker.py

# Step 3: Feature Extraction from Tracks
python feature_extractor.py

# Step 4: Matching Tracks via Cosine Similarity
python player_matcher.py

# Step 5: Generate Combined Visualization Video
python match_video_generator.py
```

---

##  Methodology

- **Detection**: Player and ball detection YOLOv11 model.
- **Tracking**: DeepSORT with MobileNet embedder for assigning player IDs across time.
- **Feature Extraction**: Mean color histograms (RGB), width, height, and duration per track.
- **Matching**: Cosine similarity on extracted features, filtered by duration and uniqueness.
- **Visualization**: Dual-view video with matched IDs annotated.

---

##  Dependencies

Installed using `requirements.txt`. Main packages are:

- `ultralytics`
- `deep-sort-realtime`
- `opencv-python`
- `torch`, `torchvision`
- `numpy`, `pandas`, `scikit-learn`
- `matplotlib`, `tqdm`

---

##  Outputs

| Output File | Description |
|-------------|-------------|
| `detections_*.csv` | YOLOv11 detections per frame |
| `tracks_*.csv` | DeepSORT-tracked player IDs |
| `features_*.csv` | Feature vectors per track |
| `player_mapping_final.csv` | Matched tacticam ↔ broadcast IDs |
| `matched_dual_view.mp4` | Side-by-side video with annotated matches |
| `matches_vis/` | Cropped visual match comparisons |

---

##  Notes

- Tracks with fewer than **30 frames** are filtered out as noise.
- Matching is one-to-one: best unique match retained per tacticam/broadcast ID.
- You can adjust the similarity threshold in `player_matcher.py`.

---

##  Author

Siddhi Patil
Email: siddhipatil064@gmail.com 
GitHub: [https://github.com/siddhi-patil-06]

---

## ???? Final Thoughts

This solution highlights modularity, simplicity, and real-world usability. The model is open-ended, while the aim is correct, consistent identity mapping under real-world camera variations.
