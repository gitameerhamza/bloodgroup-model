from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# =============================
# Load TFLite model (ONCE)
# =============================
interpreter = tf.lite.Interpreter(model_path="agglutination_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = 224

# =============================
# Helper functions
# =============================
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_agglutination(img_path):
    img = preprocess_image(img_path)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0][0]
    return float(output)

def get_blood_group(a, b, d):
    a_detected = a <= 0.65
    b_detected = b <= 0.65
    d_detected = d <= 0.65

    if a_detected and not b_detected and d_detected:
        return "A+"
    if a_detected and not b_detected and not d_detected:
        return "A−"
    if not a_detected and b_detected and d_detected:
        return "B+"
    if not a_detected and b_detected and not d_detected:
        return "B−"
    if a_detected and b_detected and d_detected:
        return "AB+"
    if a_detected and b_detected and not d_detected:
        return "AB−"
    if not a_detected and not b_detected and d_detected:
        return "O+"
    if not a_detected and not b_detected and not d_detected:
        return "O−"
    return "Unknown"

# =============================
# WEB ROUTE (unchanged)
# =============================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        required_keys = ['antiA', 'antiB', 'antiD']
        if not all(k in request.files for k in required_keys):
            return render_template('index.html', prediction_scores=None, result="Missing files", image_urls=None)

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        prediction_scores = {}
        image_urls = {}
        filepaths = {}

        for key in required_keys:
            file = request.files[key]
            if file and file.filename != '':
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                filepaths[key] = filepath
                image_urls[key] = filepath
            else:
                return render_template('index.html', prediction_scores=None, result="Invalid file", image_urls=None)

        prediction_scores['antiA'] = round(predict_agglutination(filepaths['antiA']), 4)
        prediction_scores['antiB'] = round(predict_agglutination(filepaths['antiB']), 4)
        prediction_scores['antiD'] = round(predict_agglutination(filepaths['antiD']), 4)

        result = get_blood_group(
            prediction_scores['antiA'],
            prediction_scores['antiB'],
            prediction_scores['antiD']
        )

        return render_template(
            'index.html',
            prediction_scores=prediction_scores,
            result=result,
            image_urls=image_urls
        )

    return render_template('index.html', prediction_scores=None, result=None, image_urls=None)

# =============================
# 🔥 API ROUTE (For Mobile App)
# =============================
@app.route('/api/predict', methods=['POST'])
def api_predict():
    required_keys = ['antiA', 'antiB', 'antiD']

    if not all(k in request.files for k in required_keys):
        return jsonify({"error": "Missing images"}), 400

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    scores = {}
    paths = {}

    for key in required_keys:
        file = request.files[key]
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        paths[key] = path

    scores['antiA'] = round(predict_agglutination(paths['antiA']), 4)
    scores['antiB'] = round(predict_agglutination(paths['antiB']), 4)
    scores['antiD'] = round(predict_agglutination(paths['antiD']), 4)

    blood_group = get_blood_group(
        scores['antiA'],
        scores['antiB'],
        scores['antiD']
    )

    return jsonify({
        "status": "success",
        "scores": scores,
        "blood_group": blood_group
    })

# =============================
# Run Server
# =============================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port)
