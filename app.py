import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from model import predict_disease

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'plantcare-secret-key-2025')
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Routes ──────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('upload.html')

    # POST: handle file upload + prediction
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload JPG, JPEG, PNG, or WebP.'}), 400

    # Save file with unique name
    ext = file.filename.rsplit('.', 1)[1].lower()
    unique_filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    # Run prediction
    result = predict_disease(filepath)

    # Build image URL for display
    result['image_url'] = url_for('static', filename=f'uploads/{unique_filename}')
    result['filename'] = secure_filename(file.filename)

    # Return JSON for AJAX or redirect for form submit
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or \
       request.accept_mimetypes.best == 'application/json':
        return jsonify(result)

    # Redirect to result page
    if result['is_healthy']:
        return redirect(url_for('result_healthy',
                                plant=result['plant'],
                                condition=result['condition'],
                                confidence=result['confidence'],
                                image_url=result['image_url']))
    else:
        return redirect(url_for('result_disease',
                                plant=result['plant'],
                                condition=result['condition'],
                                confidence=result['confidence'],
                                image_url=result['image_url']))


@app.route('/result')
def result():
    plant      = request.args.get('plant', 'Unknown Plant')
    condition  = request.args.get('condition', 'Unknown')
    confidence = request.args.get('confidence', '0')
    image_url  = request.args.get('image_url', '')
    is_healthy = request.args.get('is_healthy', 'false').lower() == 'true'

    from model import get_recommendations
    recommendations = get_recommendations(plant, condition, is_healthy)

    return render_template('result.html',
                           plant=plant,
                           condition=condition,
                           confidence=confidence,
                           image_url=image_url,
                           is_healthy=is_healthy,
                           recommendations=recommendations)


@app.route('/result/healthy')
def result_healthy():
    plant      = request.args.get('plant', 'Unknown Plant')
    condition  = request.args.get('condition', 'Healthy')
    confidence = request.args.get('confidence', '0')
    image_url  = request.args.get('image_url', '')

    from model import get_recommendations
    recommendations = get_recommendations(plant, condition, True)

    return render_template('healthyimage.html',
                           plant=plant,
                           condition=condition,
                           confidence=confidence,
                           image_url=image_url,
                           recommendations=recommendations)


@app.route('/result/disease')
def result_disease():
    plant      = request.args.get('plant', 'Unknown Plant')
    condition  = request.args.get('condition', 'Disease Detected')
    confidence = request.args.get('confidence', '0')
    image_url  = request.args.get('image_url', '')

    from model import get_recommendations
    recommendations = get_recommendations(plant, condition, False)

    return render_template('unhealthyimage.html',
                           plant=plant,
                           condition=condition,
                           confidence=confidence,
                           image_url=image_url,
                           recommendations=recommendations)


# ── API Endpoint (for AJAX / mobile apps) ───────────────────────────────

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    REST API endpoint.
    Accepts multipart/form-data with field 'file'.
    Returns JSON with prediction result.
    """
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid or missing file'}), 400

    ext = file.filename.rsplit('.', 1)[1].lower()
    unique_filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    result = predict_disease(filepath)
    result['recommendations'] = predict_disease.__module__ and __import__('model').get_recommendations(result['plant'], result['condition'], result['is_healthy'])
    result['success'] = True
    result['image_url'] = url_for('static', filename=f'uploads/{unique_filename}', _external=True)
    return jsonify(result)


# ── Error handlers ────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 10MB.'}), 413


@app.errorhandler(404)
def not_found(e):
    return render_template('home.html'), 404


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
