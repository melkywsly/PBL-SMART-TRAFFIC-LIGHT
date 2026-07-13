import os
import sys

# Pastikan venv site-packages selalu dipakai, apapun Python yang menjalankan file ini
_venv_site = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'venv', 'Lib', 'site-packages')
if os.path.isdir(_venv_site) and _venv_site not in sys.path:
    sys.path.insert(0, _venv_site)

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
MODEL_PATH       = os.path.join(BASE_DIR, 'models', 'best_traffic_vehicle_yolov5.pt')

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
            import yolov5
            import functools
            import pathlib

            # Model dilatih di Linux (Colab) → checkpoint menyimpan PosixPath.
            pathlib.PosixPath = pathlib.WindowsPath

            # PyTorch 2.6+ mengubah default weights_only=True yang memblokir
            # kelas custom di checkpoint YOLOv5.
            _orig_load = torch.load
            torch.load = functools.partial(_orig_load, weights_only=False)
            try:
                _model = yolov5.load(MODEL_PATH)
            finally:
                torch.load = _orig_load

            _model.conf = 0.45
            _model.iou  = 0.45
            print(f"[Model loaded] classes: {list(_model.names.values())}")
    return _model


def _run_inference(model, frame):
    """Run YOLOv5 inference and return normalised detection list.

    Returns list of (x1, y1, x2, y2, label_name, conf).
    """
    results = model(frame)
    out = []
    for *box, conf, cls in results.xyxy[0].tolist():
        x1, y1, x2, y2 = map(int, box)
        label_name = model.names[int(cls)]
        out.append((x1, y1, x2, y2, label_name, float(conf)))
    return out


# ─────────────────────────────────────────────
# LABEL → CATEGORY MAPPING
# ─────────────────────────────────────────────
LABEL_MAP = {
    'motorcycle': 'motor', 'bicycle': 'motor', 'motor': 'motor',
    'motorbike': 'motor', 'bikes': 'motor', 'scooter': 'motor',
    'sepeda_motor': 'motor', 'motor_besar': 'motor',
    'car': 'car', 'SUV': 'car', 'taxi': 'car', 'van': 'car',
    'pickup': 'car', 'minivan': 'car', 'mobil': 'car',
    'auto_rickshaw': 'car', 'kendaraan_ringan': 'car',
    'bus': 'bus', 'micro_bus': 'bus', 'school_bus': 'bus',
    'minibus': 'bus', 'angkot': 'bus', 'angkutan': 'bus',
    'truck': 'truck', 'mini_truck': 'truck', 'tempo': 'truck',
    'tractor': 'truck', 'transport_vehicle': 'truck',
    'heavy_vehicle': 'truck', 'truk': 'truck', 'kendaraan_berat': 'truck',
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
SAMPLE_INTERVAL = 3   # process every Nth frame (lower = finer tracking, less ID switching)

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


def _dedup_detections(detections, iou_thresh=0.5):
    """Collapse multiple detections that are really the same physical
    vehicle.

    Ultralytics applies NMS internally, but only *within* each class — if
    the model is unsure and emits e.g. both a 'car' and a 'truck' box for
    the same overhead vehicle, both survive NMS and arrive here as two
    separate detections. Left alone, the tracker spins up two parallel
    track chains for one real vehicle, inflating the total count even
    when the position-matching logic is working correctly. Keep only the
    highest-confidence box in each overlapping group, regardless of class.
    """
    dets = sorted(detections, key=lambda d: d[5], reverse=True)
    kept = []
    for d in dets:
        if all(_iou(d[:4], k[:4]) < iou_thresh for k in kept):
            kept.append(d)
    return kept


class SimpleTracker:
    """2-stage tracker with confirmation gate to suppress phantom IDs.

    A new track is only added to total_counted after it has been matched
    for min_confirmed consecutive sampled frames. Single-frame false
    positives (low-conf detections at frame edges, shadows, etc.) are
    silently discarded before they inflate the count.
    """

    def __init__(self,
                 iou_threshold: float = 0.10,
                 dist_threshold: float = 1.5,
                 max_lost: int = 20,
                 min_confirmed: int = 4):
        self.tracks         = {}
        self.next_id        = 1
        self.iou_threshold  = iou_threshold
        self.dist_threshold = dist_threshold
        self.max_lost       = max_lost
        self.min_confirmed  = min_confirmed
        self.total_counted  = 0
        self.cat_counts     = {'motor': 0, 'car': 0, 'bus': 0, 'truck': 0}

    def _assign(self, tid, det):
        t = self.tracks[tid]
        t['box']  = list(det[:4])
        t['cat']  = det[4]
        t['conf'] = det[5]
        t['lost'] = 0
        t['hits'] = t.get('hits', 0) + 1

        # Confirm track the first time hits reaches min_confirmed
        if not t.get('confirmed') and t['hits'] >= self.min_confirmed:
            t['confirmed'] = True
            self.total_counted += 1
            if t['cat'] in self.cat_counts:
                self.cat_counts[t['cat']] += 1

    def update(self, detections):
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
                norm = max((_box_diag(tb) + _box_diag(det[:4])) / 2.0, 25.0)
                dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5 / norm
                if dist < best_dist:
                    best_dist = dist
                    best_tid  = tid
            if best_tid is not None:
                matched_tids.add(best_tid)
                matched_dets.add(i)
                self._assign(best_tid, det)

        # ── Create tentative tracks (not counted yet) ──────────────────
        for i, det in enumerate(detections):
            if i not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    'box':       list(det[:4]),
                    'cat':       det[4],
                    'conf':      det[5],
                    'lost':      0,
                    'hits':      1,
                    'confirmed': False,
                }

        # ── Age unmatched tracks ───────────────────────────────────────
        for tid in track_ids:
            if tid not in matched_tids:
                self.tracks[tid]['lost'] += 1

        # ── Prune dead tracks ──────────────────────────────────────────
        dead = [tid for tid, t in self.tracks.items() if t['lost'] > self.max_lost]
        for tid in dead:
            del self.tracks[tid]

        # ── Return only confirmed + visible tracks ─────────────────────
        return [
            (int(t['box'][0]), int(t['box'][1]),
             int(t['box'][2]), int(t['box'][3]),
             t['cat'], t['conf'], tid)
            for tid, t in self.tracks.items()
            if t['lost'] == 0 and t.get('confirmed')
        ]



def process_lane_image(image_path, output_path):
    """Process a single still image: detect vehicles, draw boxes, save annotated image."""
    model = get_model()
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError("Cannot open image file.")

    raw = _run_inference(model, frame)

    detections = []
    for (x1, y1, x2, y2, label_name, conf) in raw:
        if conf < 0.40:
            continue
        bw, bh = x2 - x1, y2 - y1
        if bw * bh < MIN_BOX_AREA:
            continue
        if bw / max(bh, 1) < 0.6:
            continue

        category = LABEL_MAP.get(label_name)
        if not category:
            continue

        category = smart_category(category, x1, y1, x2, y2)
        detections.append((x1, y1, x2, y2, category, conf))

    # Collapse same-vehicle detections that landed in different classes.
    detections = _dedup_detections(detections)

    counts = {'motor': 0, 'car': 0, 'bus': 0, 'truck': 0}
    for i, (x1, y1, x2, y2, category, conf) in enumerate(detections):
        counts[category] += 1

        color = VEHICLE_COLORS.get(category, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_text = f"{category} #{i+1} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1 - 6, th + 4)
        cv2.rectangle(frame, (x1, label_y - th - 4), (x1 + tw + 4, label_y), color, -1)
        cv2.putText(frame, label_text, (x1 + 2, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    total = sum(counts.values())
    hud_text = f"Total Vehicles: {total}"
    (hw, hh), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame, (6, 6), (hw + 22, hh + 20), (0, 0, 0), -1)
    cv2.putText(frame, hud_text, (12, hh + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(output_path, frame)

    density = (counts['motor'] * WEIGHTS['motor'] +
               counts['car']   * WEIGHTS['car']   +
               counts['bus']   * WEIGHTS['bus']   +
               counts['truck'] * WEIGHTS['truck'])

    return {
        'counts':       dict(counts),
        'peakCounts':   dict(counts),
        'totalTracked': total,
        'density':      density,
        'duration':     get_green_duration(density),
        'frameLog': [{
            'frame': 1,
            'motor': counts['motor'], 'car': counts['car'],
            'bus':   counts['bus'],   'truck': counts['truck'],
            'total': total, 'totalTracked': total,
        }],
    }


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

    # Keep a track alive for ~4s of real video time regardless of fps,
    # so a vehicle briefly missed by detection isn't re-counted as new.
    max_lost = max(15, int(round(fps * 4 / SAMPLE_INTERVAL)))
    tracker  = SimpleTracker(iou_threshold=0.10, dist_threshold=1.5,
                             max_lost=max_lost, min_confirmed=4)
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

        # YOLOv5 inference
        raw = _run_inference(model, frame)

        # ── Build detection list for this frame ──
        detections = []
        for (x1, y1, x2, y2, label_name, conf) in raw:
            if conf < 0.50:
                continue

            bw = x2 - x1
            bh = y2 - y1

            if bw * bh < MIN_BOX_AREA:
                continue

            aspect_ratio = bw / max(bh, 1)
            if aspect_ratio < 0.8:
                continue

            category = LABEL_MAP.get(label_name)
            if not category:
                continue

            category = smart_category(category, x1, y1, x2, y2)
            detections.append((x1, y1, x2, y2, category, conf))

        # Collapse same-vehicle detections that landed in different classes
        # before they ever reach the tracker.
        detections = _dedup_detections(detections)

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


@app.route('/api/upload-lane-image', methods=['POST'])
def upload_lane_image():
    """Upload + process a single lane image.
    Form fields: jobId, lane (North|East|South|West), image (file)
    """
    job_id = request.form.get('jobId', '').strip()
    lane   = request.form.get('lane',  '').strip()

    if lane not in DIRECTIONS:
        return jsonify({'error': f'Invalid lane "{lane}". Must be one of {DIRECTIONS}.'}), 400

    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded.'}), 400

    file = request.files['image']
    if not file.filename:
        return jsonify({'error': 'Empty filename.'}), 400

    with _jobs_lock:
        if job_id not in _jobs:
            _jobs[job_id] = {
                'lanes':  {d: None for d in DIRECTIONS},
                'status': {d: 'pending' for d in DIRECTIONS},
            }
        _jobs[job_id]['status'][lane] = 'processing'

    uid      = str(uuid.uuid4())[:8]
    filename = secure_filename(file.filename)
    ext      = os.path.splitext(filename)[1].lower() or '.jpg'
    in_path  = os.path.join(UPLOAD_FOLDER,    f"{lane}_{uid}_{filename}")
    out_name = f"processed_{lane}_{uid}{ext}"
    out_path = os.path.join(PROCESSED_FOLDER, out_name)

    file.save(in_path)

    try:
        result = process_lane_image(in_path, out_path)
        result['outputImage'] = f'/processed/{out_name}'
        result['lane']        = lane

        with _jobs_lock:
            _jobs[job_id]['lanes'][lane]  = result
            _jobs[job_id]['status'][lane] = 'done'

        try:
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
        except Exception:
            pass

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
    counts  = {d: (done_lanes[d]['counts']  if d in done_lanes else {'motor': 0, 'car': 0, 'bus': 0, 'truck': 0})
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
                    'frame': fn, 'motor': 0, 'car': 0,
                    'bus': 0, 'truck': 0, 'total': 0, 'totalTracked': 0,
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
        'outputVideos':  {d: done_lanes[d].get('outputVideo') or done_lanes[d].get('outputImage') for d in done_lanes},
        'outputTypes':   {d: ('image' if 'outputImage' in done_lanes[d] else 'video') for d in done_lanes},
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
    print("Loading model, please wait...")
    get_model()
    print("Smart Traffic Dashboard (4-Lane) -- http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
