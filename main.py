"""
PPE Detection System - Main Flask Application
AI-Based Workplace Safety Monitoring
"""

import os
import cv2
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename
from datetime import datetime

from modules.image_processor import ImageProcessor
from modules.detector import PPEDetector
from modules.safety_compliance import SafetyComplianceChecker
from modules.visualizer import ResultVisualizer
from modules.database import Database

# ─── App Configuration ────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ppe-detection-secret-key'
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi'}

# ─── Module Initialization ────────────────────────────────────────────────────
image_processor = ImageProcessor()
detector        = PPEDetector(model_path='models/ppe_yolov8.pt')
compliance      = SafetyComplianceChecker()
visualizer      = ResultVisualizer()
db              = Database('ppe_detection.db')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_image(image: np.ndarray, source: str = "upload") -> dict:
    """Full pipeline: preprocess → detect → comply → visualize → log."""
    preprocessed = image_processor.preprocess(image)
    detections    = detector.detect(preprocessed)
    compliance_result = compliance.check(detections)
    annotated     = visualizer.draw(image.copy(), detections, compliance_result)

    _, buffer = cv2.imencode('.jpg', annotated)
    encoded   = base64.b64encode(buffer).decode('utf-8')

    db.log_detection(
        source=source,
        detections=detections,
        is_compliant=compliance_result['is_compliant'],
        violations=compliance_result['violations'],
    )

    return {
        'annotated_image': f'data:image/jpeg;base64,{encoded}',
        'detections': detections,
        'compliance': compliance_result,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    stats = db.get_summary_stats()
    return render_template('index.html', stats=stats)


@app.route('/dashboard')
def dashboard():
    records = db.get_recent_detections(limit=50)
    stats   = db.get_summary_stats()
    return render_template('dashboard.html', records=records, stats=stats)


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    image = cv2.imread(save_path)
    if image is None:
        return jsonify({'error': 'Could not read image'}), 400

    result = process_image(image, source="upload")
    return jsonify(result)


@app.route('/webcam', methods=['POST'])
def webcam_frame():
    """Receive a base64 frame from the browser webcam and run detection."""
    data = request.get_json()
    if not data or 'frame' not in data:
        return jsonify({'error': 'No frame data'}), 400

    try:
        frame_data  = data['frame'].split(',')[1]
        frame_bytes = base64.b64decode(frame_data)
        np_arr      = np.frombuffer(frame_bytes, dtype=np.uint8)
        image       = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        result = process_image(image, source="webcam")
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def api_stats():
    return jsonify(db.get_summary_stats())


@app.route('/api/recent')
def api_recent():
    limit = request.args.get('limit', 20, type=int)
    return jsonify(db.get_recent_detections(limit=limit))


@app.route('/api/compliance-trend')
def api_compliance_trend():
    return jsonify(db.get_compliance_trend())


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    db.init()
    app.run(debug=True, host='0.0.0.0', port=5000)
