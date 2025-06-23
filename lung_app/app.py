import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from datetime import datetime
from PIL import Image
import numpy as np

# Grad-CAM sınıfını içe aktar
from gradcam_helper import PneumoniaGradCAM

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
MARKED_FOLDER = 'static/marked'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MARKED_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MARKED_FOLDER'] = MARKED_FOLDER

MODEL_PATH = 'models/pnomoni_modeli_son.h5'
IMG_SIZE = (150, 150)

gradcam = PneumoniaGradCAM(MODEL_PATH, IMG_SIZE)

# Basit analiz geçmişi (uygulama açık kaldığı sürece)
analysis_history = []

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    percent = None
    image_url = None
    marked_url = None

    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Görüntüyü analiz et
            analysis = gradcam.analyze_image(file_path, threshold=0.5, show_plot=False)

            probability = analysis['probability']
            label = 'HASTA' if probability > 0.5 else 'SAĞLIKLI'
            confidence_percent = f"{probability*100:.2f}%" if probability > 0.5 else f"{(1-probability)*100:.2f}%"

            result = label
            percent = confidence_percent
            image_url = f'uploads/{filename}'

            marked_url = None
            if 'overlay_image' in analysis:
                overlay_img = analysis['overlay_image']  # NumPy BGR formatında
                overlay_rgb = overlay_img[..., ::-1]  # BGR'den RGB'ye dönüştür

                pil_img = Image.fromarray(overlay_rgb)
                marked_filename = f"marked_{filename}"
                marked_path = os.path.join(app.config['MARKED_FOLDER'], marked_filename)
                pil_img.save(marked_path)
                marked_url = f'marked/{marked_filename}'

            # Geçmişe kaydet
            analysis_history.append({
                'filename': filename,
                'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'result': label,
                'percent': confidence_percent,
                'image_url': image_url,
                'marked_url': marked_url
            })

    return render_template(
        'index.html',
        result=result,
        percent=percent,
        image_url=image_url,
        marked_url=marked_url,
        history=analysis_history
    )

if __name__ == '__main__':
    app.run(debug=True)
