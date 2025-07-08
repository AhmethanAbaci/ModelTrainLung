import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

xray_model = load_model('models/xray_classifier.h5')
disease_model = load_model('models/pnomoni_modeli.h5')

IMG_SIZE = (150, 150)

history = []

def preprocess(img):
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def is_xray(img):
    x = preprocess(img)
    pred = xray_model.predict(x)[0][0]
    return pred > 0.5, pred * 100

def detect_disease(img):
    x = preprocess(img)
    pred = disease_model.predict(x)[0][0]
    percent = pred * 100
    label = "HASTA" if pred > 0.5 else "SAĞLIKLI"
    return label, f"{percent:.2f}%"

@app.route('/', methods=['GET', 'POST'])
def index():
    global history

    result = None
    percent = None
    image_url = None
    xray_warning = None

    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            img = Image.open(file_path).convert('RGB')
            is_xray_img, xray_conf = is_xray(img)

            image_url = f'uploads/{filename}'

            if not is_xray_img:
                xray_warning = f"Bu görüntü bir X-ray değildir! ({xray_conf:.2f}%)"
                result = "Analiz yapılmadı"
                percent = "-"
            else:
                result, percent = detect_disease(img)
                xray_warning = f"X-ray doğrulandı. ({xray_conf:.2f}%)"
                new_entry = {
                    'filename': filename,
                    'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'result': result,
                    'percent': percent
                }
                history.append(new_entry)

    return render_template(
        'index.html',
        result=result,
        percent=percent,
        image_url=image_url,
        xray_warning=xray_warning,
        history=history
    )

@app.route('/history/<int:index>')
def show_history(index):
    global history
    if 0 <= index < len(history):
        item = history[index]
        return render_template(
            'index.html',
            result=item['result'],
            percent=item['percent'],
            image_url=f"uploads/{item['filename']}",
            xray_warning=None,
            history=history
        )
    return "Geçersiz kayıt", 404

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

if __name__ == '__main__':
    app.run(debug=True)
