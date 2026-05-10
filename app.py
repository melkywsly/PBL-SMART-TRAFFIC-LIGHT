import os
import cv2
import torch
import numpy as np
import uuid
import threading
import mysql.connector
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, request, jsonify, render_template, send_from_directory, session, redirect, url_for
from werkzeug.utils import secure_filename

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER    = os.path.join(BASE_DIR, 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')
MODEL_PATH       = os.path.join(BASE_DIR, 'models', 'yolov8n.pt')

os.makedirs(UPLOAD_FOLDER,    exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024   # 500 MB
app.secret_key = 'super-secret-traffic-key'
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'traffic_system'
}

# ─────────────────────────────────────────────
# MODEL (lazy-load once)
# ─────────────────────────────────────────────
_model      = None
_model_lock = threading.Lock()

def get_model():
    global _model
    with _model_lock:
        if _model is None:
            from ultralytics import YOLO
            # Initialize YOLOv8 model natively
            _model = YOLO(MODEL_PATH)
    return _model


# ─────────────────────────────────────────────
# LABEL → CATEGORY MAPPING
# ─────────────────────────────────────────────
LABEL_MAP = {
    # COCO dataset classes mapping
    'motorcycle': 'motor', 'bicycle': 'motor',
    'car': 'car',
    'bus': 'bus',
    'truck': 'truck',
    # Original custom dataset mappings just in case
    'bikes': 'motor', 'scooter': 'motor', 'e-rickshaw': 'motor',
    'SUV': 'car', 'taxi': 'car', 'van': 'car', 'auto_rickshaw': 'car',
    'micro_bus': 'bus', 'school_bus': 'bus',
    'mini_truck': 'truck', 'tempo': 'truck',
    'tractor': 'truck', 'transport_vehicle': 'truck',
}

WEIGHTS    = {'motor': 1, 'car': 2, 'bus': 3, 'truck': 3}
DIRECTIONS = ['North', 'East', 'South', 'West']

VEHICLE_COLORS = {
    'motor': (0, 255, 255),
    'car':   (255, 200, 0),
    'bus':   (0, 200, 255),
    'truck': (255, 80, 0),
}


# ─────────────────────────────────────────────
# GREEN DURATION RULES
# ─────────────────────────────────────────────
def get_green_duration(density):
    if density >= 40: return 60
    if density >= 25: return 45
    if density >= 12: return 30
    if density > 0:   return 20
    return 10


# ─────────────────────────────────────────────
# PER-LANE VIDEO PROCESSING
# Each lane video is processed in full-frame (no triangular ROI split).
# The whole frame = that lane's camera view.
# ─────────────────────────────────────────────
SAMPLE_INTERVAL = 5   # process every Nth frame

# ── Minimum bounding-box area (px²) to accept a detection ──
# Raised to 2000 since we're only looking for cars, which are larger.
MIN_BOX_AREA = 2000

# ── Motor reclassification guard (overhead/CCTV view) ──
# From a top-down CCTV, motorcycles appear MUCH wider than in a side-on shot.
# The old 2500 px² + aspect>0.85 rule classified every visible motor as 'car'.
# New thresholds:
#   MOTOR_MAX_AREA = 8000  (a full-body car from above is >10 000 px²)
#   aspect > 1.6           (only reclassify if bbox is clearly car-wide)
MOTOR_MAX_AREA = 8000

def smart_category(category, x1, y1, x2, y2):
    """Post-processing reclassification — tuned for highway footage.

    The user specified: "ini motor tidak ada ini semua mobil" 
    (there are no motorcycles here, these are all cars).
    We forcefully reclassify all 'motor' detections as 'car' 
    to prevent bounding box flickering and class glitches.
    """
    if category == 'motor':
        return 'car'

    return category



def _iou(a, b):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    aa = (a[2]-a[0]) * (a[3]-a[1])
    ab = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (aa + ab - inter)


# ─────────────────────────────────────────────
# ROBUST 2-STAGE OBJECT TRACKER
# Stage 1: IoU matching (handles vehicles with good overlap)
# Stage 2: centroid-distance fallback (handles blink / brief miss)
# No external dependencies — pure Python.
# ─────────────────────────────────────────────

def _centroid(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

def _box_diag(box):
    """Diagonal length of a box — used to normalise centroid distance."""
    return max(((box[2]-box[0])**2 + (box[3]-box[1])**2) ** 0.5, 1.0)


class SimpleTracker:
    """Robust 2-stage tracker that avoids double-counting on detection blinks.

    When YOLO misses a vehicle for 1-2 sampled frames (blink), the box
    position can change enough that IoU drops to zero even though it is
    the same vehicle.  A centroid-distance fallback stage catches these
    cases and re-links the detection to the existing track, keeping the
    ID stable and the cumulative count accurate.

    Tuning parameters
    -----------------
    iou_threshold     : min IoU for Stage-1 match  (0.10 — very permissive)
    dist_threshold    : max normalised centroid distance for Stage-2 match
                        (1.0 means the centroid can move up to ~1 box diagonal)
    max_lost          : sampled frames before a track is pruned
                        (20 samples × 5 frame-interval ≈ 4 s at 25 fps)
    alpha             : EMA smoothing for box position (0 = no smoothing)

    Attributes
    ----------
    total_counted : int   — cumulative unique vehicles (no double-count)
    cat_counts    : dict  — cumulative count by category
    """

    def __init__(self,
                 iou_threshold: float = 0.10,
                 dist_threshold: float = 1.0,
                 max_lost: int = 20):
        self.tracks         = {}   # tid -> {'box', 'cat', 'lost', 'conf'}
        self.next_id        = 1
        self.iou_threshold  = iou_threshold
        self.dist_threshold = dist_threshold
        self.max_lost       = max_lost
        self.total_counted  = 0
        self.cat_counts     = {'motor': 0, 'car': 0, 'bus': 0, 'truck': 0}

    def _assign(self, tid, det):
        """Link detection to track tid, updating box + resetting lost."""
        self.tracks[tid].update({
            'box':  list(det[:4]),
            'cat':  det[4],
            'conf': det[5],
            'lost': 0,
        })

    def update(self, detections):
        """Match detections to tracks; return currently-visible tracked objects.

        Parameters
        ----------
        detections : list of (x1, y1, x2, y2, category, conf)

        Returns
        -------
        list of (x1, y1, x2, y2, category, conf, track_id)
        """
        track_ids    = list(self.tracks.keys())
        matched_tids = set()
        matched_dets = set()

        # ── Stage 1: IoU matching ──────────────────────────────────────
        for i, det in enumerate(detections):
            best_score = self.iou_threshold
            best_tid   = None
            for tid in track_ids:
                if tid in matched_tids:
                    continue
                score = _iou(det[:4], self.tracks[tid]['box'])
                if score > best_score:
                    best_score = score
                    best_tid   = tid
            if best_tid is not None:
                matched_tids.add(best_tid)
                matched_dets.add(i)
                self._assign(best_tid, det)

        # ── Stage 2: centroid-distance fallback ───────────────────────
        # For unmatched detections, try to link to an unmatched track
        # via normalised centroid distance (handles blink / partial miss).
        remaining_tids = [tid for tid in track_ids if tid not in matched_tids]
        for i, det in enumerate(detections):
            if i in matched_dets:
                continue
            cx, cy = _centroid(det[:4])
            best_dist = self.dist_threshold
            best_tid  = None
            for tid in remaining_tids:
                if tid in matched_tids:
                    continue
                tb = self.tracks[tid]['box']
                tx, ty = _centroid(tb)
                dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5 / _box_diag(tb)
                if dist < best_dist:
                    best_dist = dist
                    best_tid  = tid
            if best_tid is not None:
                matched_tids.add(best_tid)
                matched_dets.add(i)
                self._assign(best_tid, det)

        # ── Create new tracks for genuinely new vehicles ───────────────
        for i, det in enumerate(detections):
            if i not in matched_dets:
                tid = self.next_id
                self.next_id       += 1
                self.total_counted += 1
                cat = det[4]
                if cat in self.cat_counts:
                    self.cat_counts[cat] += 1
                self.tracks[tid] = {
                    'box':  list(det[:4]),
                    'cat':  cat,
                    'conf': det[5],
                    'lost': 0,
                }

        # ── Age unmatched tracks ───────────────────────────────────────
        for tid in track_ids:
            if tid not in matched_tids:
                self.tracks[tid]['lost'] += 1

        # ── Prune tracks dead for too long ─────────────────────────────
        dead = [tid for tid, t in self.tracks.items() if t['lost'] > self.max_lost]
        for tid in dead:
            del self.tracks[tid]

        # ── Return visible tracks (lost == 0) ─────────────────────────
        return [
            (int(t['box'][0]), int(t['box'][1]),
             int(t['box'][2]), int(t['box'][3]),
             t['cat'], t['conf'], tid)
            for tid, t in self.tracks.items() if t['lost'] == 0
        ]



def process_lane_video(video_path, output_path):
    """Process a single-lane video with IoU-based object tracking.

    Each detected vehicle gets a persistent unique ID drawn next to its
    bounding box (e.g. "Car #1", "Bus #3").  A HUD overlay on every
    frame shows the running total of unique vehicles seen so far.

    Counting strategy
    -----------------
    * tracker.total_counted  — cumulative unique IDs ever assigned
      (= vehicles that have *passed through* the camera, no double-counting)
    * peak_counts            — max concurrent vehicles per category in any
      single frame  (used only for density / adaptive signal timing)
    """
    model = get_model()
    cap   = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, max(fps / SAMPLE_INTERVAL, 5), (w, h))

    tracker     = SimpleTracker(iou_threshold=0.10, dist_threshold=1.0, max_lost=20)
    peak_counts = {'motor': 0, 'car': 0, 'bus': 0, 'truck': 0}
    frame_log   = []
    frame_num   = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        if frame_num % SAMPLE_INTERVAL != 0:
            continue

        # YOLOv8 inference
        results = model(frame, verbose=False)
        names   = model.names

        # ── Build detection list for this frame ──
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls  = int(box.cls[0])

            # Apply hard confidence threshold (simulating previous model.conf = 0.35)
            if conf < 0.35:
                continue

            bw = x2 - x1
            bh = y2 - y1

            if bw * bh < MIN_BOX_AREA:
                continue
                
            # Discard tall/skinny bounding boxes (aspect ratio < 0.6) 
            # These are typically false positives like dashed road lines.
            aspect_ratio = bw / max(bh, 1)
            if aspect_ratio < 0.6:
                continue

            label_name = names[cls]
            category   = LABEL_MAP.get(label_name)
            if not category:
                continue

            category = smart_category(category, x1, y1, x2, y2)
            detections.append((x1, y1, x2, y2, category, conf))

        # ── Update tracker ──
        # Pass full detections including conf to tracker
        tracked = tracker.update(detections)

        # Per-frame concurrent counts (for peak detection / density)
        frame_counts = {'motor': 0, 'car': 0, 'bus': 0, 'truck': 0}

        # ── Draw tracked bounding boxes with persistent IDs ──
        for (x1, y1, x2, y2, category, conf, track_id) in tracked:
            frame_counts[category] += 1
            color = VEHICLE_COLORS.get(category, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Solid label background for readability, exactly like original requested UI
            label_text = f"{category} #{track_id} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y1 - 6, th + 4)
            cv2.rectangle(frame, (x1, label_y - th - 4), (x1 + tw + 4, label_y), color, -1)
            cv2.putText(frame, label_text, (x1 + 2, label_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # ── HUD overlay: total unique tracked + active this frame ──
        total_tracked = tracker.total_counted
        active_now    = sum(frame_counts.values())
        hud_text      = f"Total Vehicles: {total_tracked}   Active: {active_now}"
        (hw, hh), _   = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(frame, (6, 6), (hw + 22, hh + 20), (0, 0, 0), -1)
        cv2.putText(frame, hud_text, (12, hh + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)

        # Update peak concurrent counts
        for cat in peak_counts:
            if frame_counts[cat] > peak_counts[cat]:
                peak_counts[cat] = frame_counts[cat]

        frame_total = sum(frame_counts.values())
        if len(frame_log) < 50:
            frame_log.append({
                'frame':        frame_num,
                'total':        frame_total,
                'totalTracked': tracker.total_counted,
                **frame_counts,
            })

    cap.release()
    out.release()

    # Density uses peak concurrent counts (representative of road load)
    density = (peak_counts['motor'] * WEIGHTS['motor'] +
               peak_counts['car']   * WEIGHTS['car']   +
               peak_counts['bus']   * WEIGHTS['bus']   +
               peak_counts['truck'] * WEIGHTS['truck'])

    return {
        'counts':       dict(tracker.cat_counts),   # unique vehicles by type
        'peakCounts':   peak_counts,                # peak concurrent (density only)
        'totalTracked': tracker.total_counted,      # grand total unique IDs
        'density':      density,
        'duration':     get_green_duration(density),
        'frameLog':     frame_log,
    }


# ─────────────────────────────────────────────
# IN-MEMORY JOB STORE  (for polling results)
# ─────────────────────────────────────────────
_jobs = {}   # job_id → {'lanes': {lane: result|None}, 'status': {lane: 'pending'|'done'|'error'}}
_jobs_lock = threading.Lock()


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route('/')
def index():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session.get('username', 'Admin'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT id, password_hash FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()
            conn.close()
            
            if user and check_password_hash(user['password_hash'], password):
                session['logged_in'] = True
                session['username'] = username
                return redirect(url_for('index'))
            else:
                return render_template('login.html', error='Invalid username or password')
        except Exception as e:
            # Development mode: allow login if MySQL unavailable
            if username and password:
                session['logged_in'] = True
                session['username'] = username
                return redirect(url_for('index'))
            return render_template('login.html', error='Database unavailable. Please enter any username/password to continue (dev mode).')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/api/new-job', methods=['POST'])
def new_job():
    """Create a fresh job ID for a new intersection session."""
    job_id = str(uuid.uuid4())[:12]
    with _jobs_lock:
        _jobs[job_id] = {
            'lanes':  {d: None for d in DIRECTIONS},
            'status': {d: 'pending' for d in DIRECTIONS},
        }
    return jsonify({'jobId': job_id})


@app.route('/api/upload-lane', methods=['POST'])
def upload_lane():
    """Upload + process a single lane video.
    Form fields: jobId, lane (North|East|South|West), video (file)
    """
    job_id = request.form.get('jobId', '').strip()
    lane   = request.form.get('lane',  '').strip()

    if lane not in DIRECTIONS:
        return jsonify({'error': f'Invalid lane "{lane}". Must be one of {DIRECTIONS}.'}), 400

    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded.'}), 400

    file = request.files['video']
    if not file.filename:
        return jsonify({'error': 'Empty filename.'}), 400

    # Ensure job exists (create ad-hoc if client skipped /api/new-job)
    with _jobs_lock:
        if job_id not in _jobs:
            _jobs[job_id] = {
                'lanes':  {d: None for d in DIRECTIONS},
                'status': {d: 'pending' for d in DIRECTIONS},
            }
        _jobs[job_id]['status'][lane] = 'processing'

    uid      = str(uuid.uuid4())[:8]
    filename = secure_filename(file.filename)
    in_path  = os.path.join(UPLOAD_FOLDER,   f"{lane}_{uid}_{filename}")
    out_name = f"processed_{lane}_{uid}.mp4"
    out_path = os.path.join(PROCESSED_FOLDER, out_name)

    file.save(in_path)

    try:
        result = process_lane_video(in_path, out_path)
        result['outputVideo'] = f'/processed/{out_name}'
        result['lane']        = lane

        with _jobs_lock:
            _jobs[job_id]['lanes'][lane]  = result
            _jobs[job_id]['status'][lane] = 'done'

        # --- INSERT INTO DB ---
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO traffic_logs 
                          (job_id, lane, motor_count, car_count, bus_count, truck_count, density_score, green_duration)
                          VALUES (%s, %s, %s, %s, %s, %s, %s, %s)''',
                       (job_id, lane, 
                        int(result['counts'].get('motor', 0)), 
                        int(result['counts'].get('car', 0)),
                        int(result['counts'].get('bus', 0)),
                        int(result['counts'].get('truck', 0)),
                        float(result['density']),
                        int(result['duration'])))
        conn.commit()
        conn.close()

        return jsonify(result)

    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]['status'][lane] = 'error'
        return jsonify({'error': str(e)}), 500


@app.route('/api/results/<job_id>', methods=['GET'])
def get_results(job_id):
    """Return combined 4-lane results for a job.
    Computes adaptive signal allocation across all completed lanes.
    """
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        return jsonify({'error': 'Job not found.'}), 404

    # Only use lanes that finished
    done_lanes = {d: job['lanes'][d] for d in DIRECTIONS if job['status'][d] == 'done'}

    if not done_lanes:
        return jsonify({'status': 'pending', 'message': 'No lanes processed yet.'}), 202

    density = {d: (done_lanes[d]['density'] if d in done_lanes else 0) for d in DIRECTIONS}
    counts  = {d: (done_lanes[d]['counts']  if d in done_lanes else {'motor':0,'car':0,'bus':0,'truck':0})
               for d in DIRECTIONS}

    # Adaptive signal: lane with highest density gets GREEN
    green    = max(density, key=density.get) if any(density.values()) else 'North'
    duration = get_green_duration(density[green])

    # Sum cumulative unique tracked vehicles across all completed lanes
    total = sum(done_lanes[d].get('totalTracked', 0) for d in done_lanes)

    # Sum per vehicle-type across all completed lanes
    total_by_type = {'motor': 0, 'car': 0, 'bus': 0, 'truck': 0}
    for d, res in done_lanes.items():
        lane_counts = res.get('counts', {})
        for vtype in total_by_type:
            total_by_type[vtype] += lane_counts.get(vtype, 0)

    # Build decision log from all lanes (merge by frame number)
    # Use per-type format so the frontend log table renders correctly.
    combined_log = {}
    for d, res in done_lanes.items():
        for entry in res.get('frameLog', []):
            fn = entry['frame']
            if fn not in combined_log:
                combined_log[fn] = {
                    'frame':        fn,
                    'motor':        0,
                    'car':          0,
                    'bus':          0,
                    'truck':        0,
                    'total':        0,
                    'totalTracked': 0,
                }
            combined_log[fn]['motor']        += entry.get('motor', 0)
            combined_log[fn]['car']          += entry.get('car',   0)
            combined_log[fn]['bus']          += entry.get('bus',   0)
            combined_log[fn]['truck']        += entry.get('truck', 0)
            combined_log[fn]['total']        += entry.get('total', 0)
            combined_log[fn]['totalTracked'] += entry.get('totalTracked', 0)

    decision_log = sorted(combined_log.values(), key=lambda x: x['frame'])[:50]

    return jsonify({
        'jobStatus':     job['status'],
        'counts':        counts,
        'density':       density,
        'green':         green,
        'duration':      duration,
        'totalVehicles': total,
        'totalByType':   total_by_type,
        'decisionLog':   decision_log,
        'outputVideos':  {d: done_lanes[d]['outputVideo'] for d in done_lanes},
    })


@app.route('/api/job-status/<job_id>', methods=['GET'])
def job_status(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found.'}), 404
    return jsonify({'status': job['status']})


@app.route('/processed/<path:filename>')
def serve_processed(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)


@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    print("Smart Traffic Dashboard (4-Lane) -- http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
