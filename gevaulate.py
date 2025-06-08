import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- Parametreler ---
model_path = 'efficientnetb3_grayscale_finetuned_model.keras'
data_dir = './tdatasetaugmented'
img_height = 300
img_width = 300
batch_size = 32
validation_split_ratio = 0.2
random_seed = 123

# --- Validation verisini yükle (RGB formatında geliyor) ---
val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=validation_split_ratio,
    subset="validation",
    seed=random_seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = val_ds_raw.class_names

# --- Grayscale'e çevirme ve preprocess işlemi ---
def convert_to_grayscale_then_preprocess(images, labels):
    gray_images = tf.image.rgb_to_grayscale(images)               # (B, H, W, 1)
    rgb_images = tf.image.grayscale_to_rgb(gray_images)           # (B, H, W, 3)
    return preprocess_input(rgb_images), labels

AUTOTUNE = tf.data.AUTOTUNE
val_ds = val_ds_raw.map(convert_to_grayscale_then_preprocess).cache().prefetch(AUTOTUNE)

# --- Modeli yükle ---
model = tf.keras.models.load_model(model_path)

# --- Tahmin yap ---
true_labels = []
predicted_probs = []

for images, labels in val_ds:
    true_labels.extend(labels.numpy())
    preds = model.predict(images, verbose=0)
    predicted_probs.extend(preds)

true_labels = np.array(true_labels)
predicted_probs = np.array(predicted_probs)
predicted_labels = np.argmax(predicted_probs, axis=1)

# --- Metrikler ---
print("\n📊 Metrikler:")
print(f"Accuracy:  {accuracy_score(true_labels, predicted_labels):.4f}")
print(f"Precision: {precision_score(true_labels, predicted_labels, average='weighted', zero_division=0):.4f}")
print(f"Recall:    {recall_score(true_labels, predicted_labels, average='weighted', zero_division=0):.4f}")
print(f"F1-Score:  {f1_score(true_labels, predicted_labels, average='weighted', zero_division=0):.4f}")

# --- Sınıflandırma Raporu ---
print("\n🧾 Sınıflandırma Raporu:")
print(classification_report(true_labels, predicted_labels, target_names=class_names))

# --- Konfüzyon Matrisi ---
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(max(10, len(class_names)//2), max(10, len(class_names)//2)))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Konfüzyon Matrisi")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.tight_layout()
plt.show()
