import cv2
import time
import os
from ultralytics import YOLO
from deep_sort_tracker import get_tracker
from face_detector import load_face_detector
from face_reid import FaceReID
from flask import Flask, request, send_from_directory, jsonify

# --- Initialize Flask App ---
app = Flask(__name__, static_folder='.', static_url_path='')
@app.route('/')
def index():
    # This route will serve your main HTML file
    return send_from_directory('.', 'index.html')
# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

print("[INFO] Loading models...")
# --- Load models with GPU ---
model = YOLO("yolov8n.pt").to("cuda")
tracker = get_tracker()
face_app = load_face_detector()
face_reid = FaceReID()
print("[INFO] Models loaded successfully.")

def process_video(input_path, output_path):
    """
    This function contains your original video processing logic.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {input_path}")
        return

    width = 640
    height = 360
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()
    
    print("[INFO] Starting video processing...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (width, height))
        frame_count += 1

        # --- Your processing logic ---
        if frame_count % 2 != 0:
            continue

        results = model(frame, verbose=False)[0]
        detections = []
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, cls = r
            if int(cls) == 0:
                detections.append(([x1, y1, x2 - x1, y2 - y1], score, 'person'))

        tracks = tracker.update_tracks(detections, frame=frame)
        active_ids = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            active_ids.append(track_id)
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f"Person ID: {track_id}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if frame_count % 5 == 0:
            faces = face_app.get(frame)
            for face in faces:
                bbox = face.bbox.astype(int)
                emb = face.normed_embedding
                face_id = face_reid.get_id(emb)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                cv2.putText(frame, f"Face ID: {face_id}", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        cv2.putText(frame, f"Active Persons: {len(active_ids)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Write frame to output
        out.write(frame)

    # --- Release resources ---
    cap.release()
    out.release()
    print(f"✅ Done! Output saved as '{output_path}'")

@app.route('/process-video', methods=['POST'])
def handle_video_processing():
    """
    This is the API endpoint. The frontend will send the video here.
    """
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    input_filename = f"{time.time()}_{file.filename}"
    input_path = os.path.join(UPLOAD_FOLDER, input_filename)
    file.save(input_path)

    # Define the output path
    output_filename = f"processed_{input_filename}"
    output_path = os.path.join(PROCESSED_FOLDER, output_filename)

    # Process the video
    process_video(input_path, output_path)

    # Send the processed video back to the frontend
    # NEW, CORRECT LINE
    return send_from_directory(PROCESSED_FOLDER, output_filename)

if __name__ == '__main__':
    # This makes the server accessible on your local network
    app.run(host='0.0.0.0', port=5000, debug=True)