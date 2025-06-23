import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)

# Analiz geçmişi - uygulama açık kaldığı sürece tutulacak
analysis_history = []

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model yükle
model = load_model('models/pnomoni_modeli.h5')
IMG_SIZE = (150, 150)

def preprocess(img):
    # Görüntüyü yeniden boyutlandır ve normalize et
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def detect_disease(img):
    x = preprocess(img)
    pred = model.predict(x)[0][0]
    percent = pred * 100
    label = "HASTA" if pred > 0.5 else "SAĞLIKLI"
    return label, f"{percent:.2f}%"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    percent = None
    image_url = None

    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Görüntüyü RGB'ye çevir
            img = Image.open(file_path).convert('RGB')
            result, percent = detect_disease(img)

            image_url = f'uploads/{filename}'

            # Analiz geçmişine kayıt ekle
            analysis_history.append({
                'filename': filename,
                'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'result': result,
                'percent': percent
            })

    return render_template(
        'index.html',
        result=result,
        percent=percent,
        image_url=image_url,
        history=analysis_history
    )

if __name__ == '__main__':
    app.run(debug=True)
