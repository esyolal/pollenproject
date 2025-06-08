import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

# --- Modeli Yükle ---
model = tf.keras.models.load_model("efficientnetb3_grayscale_finetuned_model.keras")

# --- Sınıf İsimlerini Yükle ---
with open("class_names2.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]

# --- Tahmin Fonksiyonu ---
def predict(img):
    img = img.resize((300, 300)).convert("L")             # Grayscale çevir
    img = img.convert("RGB")                              # Tek kanaldan tekrar 3 kanala (model uyumu için)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)               # EfficientNet preprocess

    preds = model.predict(img_array, verbose=0)[0]
    top5_indices = preds.argsort()[-5:][::-1]
    results = {class_names[i]: float(preds[i]) for i in top5_indices}
    return results

# --- Gradio Arayüzü ---
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="🌿 Polen Sınıflandırıcı (Grayscale, EfficientNetB3)",
    description="Grayscale görüntülerle eğitilmiş EfficientNetB3 modeliyle polen türlerini tahmin eder (ilk 5 sonucu gösterir)."
)

# --- Arayüzü Başlat ---
iface.launch()