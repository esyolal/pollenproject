import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt
import os

# --- Parametreler ---
model_path = 'efficientnetb3_grayscale_finetuned_model.keras'
img_path = 'z.jpeg'
img_size = (300, 300)

# --- Modeli Yükle ---
model = tf.keras.models.load_model(model_path)
class_names = sorted(os.listdir('./tdatasetaugmented'))

# --- Görseli Hazırla ---
img = image.load_img(img_path, target_size=img_size)
img_array = image.img_to_array(img)

# ➤ Grayscale'e çevir → RGB'ye geri döndür
img_gray = tf.image.rgb_to_grayscale(img_array)
img_rgb = tf.image.grayscale_to_rgb(img_gray)

# ➤ Preprocess + batch dimension ekle
img_rgb = preprocess_input(img_rgb)
img_rgb = np.expand_dims(img_rgb, axis=0)

# --- Tahmin Yap ---
preds = model.predict(img_rgb)[0]
top_5_indices = preds.argsort()[-5:][::-1]
top_5_classes = [(class_names[i], preds[i]) for i in top_5_indices]

# --- Sonucu Göster ---
print("🔍 En Yüksek 5 Tahmin:")
for i, (cls, conf) in enumerate(top_5_classes, 1):
    print(f"{i}. {cls} ({conf:.2%})")

# Görseli göster (orijinal haliyle)
plt.imshow(img)
plt.title(f"1. {top_5_classes[0][0]} ({top_5_classes[0][1]:.2%})")
plt.axis('off')
plt.show()
