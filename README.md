# 🎥 Real-Time Video Analytics System with Person Tracking and Re-Identification

This project is a fully functional, open-source **Real-Time Video Analytics System** that detects, tracks, and re-identifies individuals across frames in any input video. It assigns unique IDs, handles occlusions and reappearances, and is generalizable to any environment — **no paid APIs, no cloud services, and no special hardware dependencies**.

---

## ✨ Features

- 🧍 Person detection using **YOLOv8**
- 🧠 Person Re-Identification using **OSNet (Torchreid)**
- 🛰️ Tracking using **Deep SORT**
- 🧩 Real-time ID assignment and recovery
- 📸 Cropped face/body image display
- 📊 Statistics: Total people detected, re-entries, current people, etc.
- 💾 Optional video export with overlays
- ✅ 100% Open-source (no paid models, APIs, or services)

---

## 📌 Demo

![Demo GIF or Image Placeholder](demo/demo.gif)

---

## 📁 Project Structure

```
realtime-video-analytics/
├── detectors/
│   └── yolov8_detector.py
├── trackers/
│   └── deep_sort_tracker.py
├── reid/
│   └── osnet_reid.py
├── utils/
│   └── draw_utils.py
├── main.py
├── requirements.txt
├── README.md
└── test_videos/
```

---

## 🚀 How It Works (Simplified)

1. **YOLOv8** detects people in each video frame.
2. **Deep SORT** tracks those people and gives each a unique ID.
3. **OSNet** extracts a body-based "fingerprint" for each person.
4. If someone reappears later, **OSNet** identifies them again.
5. **OpenCV** draws overlays, shows cropped images, and stats.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/realtime-video-analytics.git
cd realtime-video-analytics
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Model Weights

- **YOLOv8** (choose one):
  - Download from: https://github.com/ultralytics/ultralytics
  - Example: `yolov8n.pt` (fast) or `yolov8s.pt` (more accurate)

- **OSNet**:
  - Install Torchreid:
    ```bash
    pip install torchreid
    ```
  - Pretrained model will be auto-downloaded on first run.

---

## 🎮 Run the System

### From Webcam:
```bash
python main.py --source 0
```

### From a Video File:
```bash
python main.py --source test_videos/sample.mp4
```

Optional flags:
- `--save`: Save output video
- `--show`: Display OpenCV window

---

## 📊 Output Display

- Video with bounding boxes and IDs
- Cropped body thumbnails per person
- Stats Panel:
  - ✅ Total people detected
  - 🔄 Re-identified people
  - 🟢 Currently visible

---

## 🧠 Models Used

| Task | Model | Why? |
|------|-------|------|
| Person Detection | YOLOv8 | Fast, accurate, lightweight |
| Tracking | Deep SORT | Popular, reliable object tracker |
| Re-Identification | OSNet (via Torchreid) | Best-in-class Re-ID without face detection |

---

## 🆚 OSNet vs Face-based Re-ID

| Feature | OSNet | FaceNet |
|---------|-------|---------|
| Works without face? | ✅ Yes | ❌ No |
| Handles occlusion? | ✅ Yes | ⚠️ Limited |
| Better for side/rear view? | ✅ Yes | ❌ No |
| Real-world use cases | ✅ Excellent | ⚠️ Only when face visible |

---

## 📦 Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics
- Torchreid
- NumPy
- FFmpeg (for export)

Install them via `requirements.txt`.

---

## 📌 Future Improvements

- [ ] Add face-based recognition as optional module
- [ ] Add web dashboard using Streamlit or Flask
- [ ] Improve UI and layout overlays
- [ ] Add entry/exit zone detection

---

## 🤝 Contributing

Pull requests are welcome! If you'd like to suggest improvements, feel free to open an issue or fork the repo.

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

- [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
- [Torchreid (OSNet)](https://github.com/KaiyangZhou/deep-person-reid)
- [Deep SORT](https://github.com/nwojke/deep_sort)

---

## 📬 Contact

For feedback or collaboration:

**Name:** AFRAH M 
**Email:** learnerafrah313@gmail.com
**GitHub:** [AFRAH-AIML](https://github.com/AFRAH-AIML)
