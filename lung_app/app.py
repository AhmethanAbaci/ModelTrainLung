from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

app = Flask(__name__)

# Modeli yükle (model yolunu projene göre ayarla)
model = load_model('models/pnomoni_modeli.h5')

IMG_SIZE = (150, 150)

def preprocess(img):
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def detect_disease(img):
    x = preprocess(img)
    pred = model.predict(x)[0][0]  # sigmoid çıkışı
    percent = pred
    label = "HASTA" if pred > 0.5 else "SAĞLIKLI"
    return label, percent


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    percent = None
    if request.method == 'POST':
        file = request.files['image']
        img = Image.open(file).convert('RGB')
        result, percent = detect_disease(img)
    return render_template('index.html', result=result, percent=percent)


if __name__ == '__main__':
    app.run(debug=True)
