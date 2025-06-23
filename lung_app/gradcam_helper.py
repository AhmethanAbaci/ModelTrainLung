import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

class PneumoniaGradCAM:
    def __init__(self, model_path, img_size=(150, 150)):
        self.img_size = img_size
        self.model_path = model_path
        self.model = None
        self.grad_model = None
        
        self._load_and_rebuild_model()
    
    def _load_and_rebuild_model(self):
        # Eski modeli yükle (model_path doğru ve model uyumlu olmalı)
        old_model = tf.keras.models.load_model(self.model_path, compile=False)
        
        inputs = tf.keras.layers.Input(shape=(*self.img_size, 1))
        
        # Model mimarisi, filtre sayıları eski modelle birebir uyumlu olmalı
        x = tf.keras.layers.Conv2D(32, (3,3), strides=1, padding='same', activation='relu')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D((2,2), strides=2, padding='same')(x)
        
        x = tf.keras.layers.Conv2D(64, (3,3), strides=1, padding='same', activation='relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D((2,2), strides=2, padding='same')(x)
        
        x = tf.keras.layers.Conv2D(64, (3,3), strides=1, padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D((2,2), strides=2, padding='same')(x)
        
        x = tf.keras.layers.Conv2D(128, (3,3), strides=1, padding='same', activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D((2,2), strides=2, padding='same')(x)
        
        last_conv = tf.keras.layers.Conv2D(256, (3,3), strides=1, padding='same', activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(last_conv)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D((2,2), strides=2, padding='same')(x)
        
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        
        # Ağırlıkları eski modelden yükle (burada hata almamalısın)
        self.model.set_weights(old_model.get_weights())
        
        # Grad-CAM için son Conv katman ve çıktı
        self.grad_model = tf.keras.models.Model(
            inputs=inputs,
            outputs=[last_conv, outputs]
        )
        
        print("Model başarıyla yüklendi ve Grad-CAM için hazırlandı!")
    
    def load_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Görüntü yüklenemedi: {image_path}")
        
        orig = img.copy()
        img = cv2.resize(img, self.img_size) / 255.0
        
        img = img.astype("float32")
        img = np.expand_dims(img, axis=-1)  # Kanal ekle
        img = np.expand_dims(img, axis=0)   # Batch ekle
        return img, orig
    
    def predict(self, image_tensor):
        return self.model.predict(image_tensor, verbose=0)[0, 0]
    
    def generate_heatmap(self, image_tensor):
        with tf.GradientTape() as tape:
            conv_out, pred = self.grad_model(image_tensor)
            class_score = pred[:, 0]
        
        grads = tape.gradient(class_score, conv_out)
        grads = tf.nn.relu(grads)  # Pozitif gradyanları al
        
        weights = tf.reduce_mean(grads, axis=(1, 2))  # Kanal bazlı ağırlıklar
        
        cam = tf.reduce_sum(weights[:, None, None, :] * conv_out, axis=-1)[0]
        
        cam = cam - tf.reduce_min(cam)
        cam = cam / (tf.reduce_max(cam) + 1e-8)
        
        return cam.numpy()
    
    def overlay_heatmap(self, heatmap, original_image, alpha=0.3):
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        original_color = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(heatmap_color, alpha, original_color, 1-alpha, 0)
    
    def analyze_image(self, image_path, threshold=0.5, show_plot=True, save_path=None):
        img_tensor, orig_img = self.load_image(image_path)
        
        probability = self.predict(img_tensor)
        
        result = {
            'probability': probability,
            'prediction': 'Zatürre' if probability > threshold else 'Normal',
            'confidence': probability if probability > threshold else 1 - probability
        }
        
        print(f"Zatürre olasılığı: {probability:.3f}")
        print(f"Tahmin: {result['prediction']} (Güven: {result['confidence']:.3f})")
        
        if probability > threshold:
            heatmap = self.generate_heatmap(img_tensor)
            overlay_img = self.overlay_heatmap(heatmap, orig_img, alpha=0.3)
            
            if show_plot:
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(orig_img, cmap='gray')
                plt.title('Orijinal Röntgen')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(heatmap, cmap='jet')
                plt.title('Grad-CAM Isı Haritası')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(overlay_img[..., ::-1])
                plt.title(f'Zatürre Odakları (Olasılık: {probability:.3f})')
                plt.axis('off')
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"Sonuç kaydedildi: {save_path}")
                plt.show()
            
            result['heatmap'] = heatmap
            result['overlay_image'] = overlay_img
        else:
            if show_plot:
                plt.figure(figsize=(8, 6))
                plt.imshow(orig_img, cmap='gray')
                plt.title(f'Normal Röntgen (Zatürre olasılığı: {probability:.3f})')
                plt.axis('off')
                plt.show()
        
        return result
