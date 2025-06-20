import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
import os

def create_model():
    model = Sequential([
        Flatten(input_shape=(224, 224, 3)),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')  # 2 sınıf çıkışı
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Dummy eğitim verisi (10 örnek)
X_dummy = np.random.rand(10, 224, 224, 3)
y_dummy = np.random.randint(0, 2, 10)

model1 = create_model()
model2 = create_model()

# Hızlıca 1 epoch ile eğit
model1.fit(X_dummy, y_dummy, epochs=1, verbose=1)
model2.fit(X_dummy, y_dummy, epochs=1, verbose=1)

# models klasörü varsa yoksa oluştur
os.makedirs("models", exist_ok=True)

model1.save("models/model1_lung.h5")
model2.save("models/model2_disease.h5")

print("Dummy Keras modelleri başarıyla oluşturuldu!")
