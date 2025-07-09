# 🫁 Zatürre Tespit Web Uygulaması | Pneumonia Detection Web App

## 🇹🇷 Proje Açıklaması

Bu proje, göğüs röntgeni (X-ray) görüntülerinden **zatürre (pneumonia)** tespiti yapabilen bir web uygulamasıdır. Kullanıcı, web arayüzü üzerinden bir görsel yükler; sistem önce bu görselin bir **X-ray** olup olmadığını kontrol eder, ardından gerçek bir X-ray ise **zatürre tanısı** koyar.

İki yapay zekâ modeli kullanılmıştır:

- `xray_model`: Görüntünün X-ray olup olmadığını tahmin eder.
- `disease_model`: X-ray görüntüsünde zatürre var mı yok mu karar verir.

### 🔧 Kullanılan Teknolojiler

- Python
- Flask
- TensorFlow / Keras
- Pillow (PIL)
- NumPy
- HTML / CSS (Jinja2 ile)

---

## 🇬🇧 Project Description

This is a web-based application that detects **pneumonia** from chest X-ray images. The user uploads an image via a browser, and the system:

1. Checks if the image is an **X-ray** using a deep learning model.
2. If it is, performs **pneumonia classification** with another model.

Two AI models are used:

- `xray_model`: Classifies whether the image is a chest X-ray.
- `disease_model`: Predicts whether the image shows signs of pneumonia.

---

## 🖥️ Uygulama Özellikleri / Features

- Web arayüzü üzerinden görsel yükleme
- Görüntünün X-ray olup olmadığını doğrulama
- X-ray görüntüsünde zatürre tespiti
- Tahmin yüzdesi gösterimi
- Önceki sonuçların geçmişte listelenmesi

---

## 📁 Kurulum / Installation

1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
