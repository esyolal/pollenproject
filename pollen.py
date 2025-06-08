import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

from tensorflow.keras.applications.efficientnet import EfficientNetB3, preprocess_input

# --- 1. Parametreler ---
data_dir = './tdatasetaugmented'
img_height = 300
img_width = 300
batch_size = 32
validation_split_ratio = 0.2
random_seed = 123
model_save_path = 'efficientnetb3_polen_model.keras'

# --- 2. Veri Yükleme ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=validation_split_ratio,
    subset="training",
    seed=random_seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=validation_split_ratio,
    subset="validation",
    seed=random_seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
class_names = train_ds.class_names

# --- 3. Pipeline ve Normalizasyon ---
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y)).cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y)).cache().prefetch(AUTOTUNE)

# --- 4. Model Mimarisi ---
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.2),
])

base_model = EfficientNetB3(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = keras.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = keras.Model(inputs, outputs)

# --- 5. Derleme ---
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- 6. Eğitim ---
epochs = 20
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# --- 7. Model Kaydetme ---
model.save(model_save_path)
print(f"\n✅ Model başarıyla kaydedildi: {model_save_path}")

# --- 8. Değerlendirme ---
true_labels = []
predicted_probs = []

for images, labels in val_ds:
    true_labels.extend(labels.numpy())
    preds = model.predict(images)
    predicted_probs.extend(preds)

true_labels = np.array(true_labels)
predicted_probs = np.array(predicted_probs)
predicted_labels = np.argmax(predicted_probs, axis=1)

print("\n--- Metrikler ---")
print(f"Accuracy: {accuracy_score(true_labels, predicted_labels):.4f}")
print(f"Precision: {precision_score(true_labels, predicted_labels, average='weighted', zero_division=0):.4f}")
print(f"Recall: {recall_score(true_labels, predicted_labels, average='weighted', zero_division=0):.4f}")
print(f"F1-Score: {f1_score(true_labels, predicted_labels, average='weighted', zero_division=0):.4f}")

print("\n--- Sınıflandırma Raporu ---")
print(classification_report(true_labels, predicted_labels, target_names=class_names))

cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(max(10, len(class_names)//2), max(10, len(class_names)//2)))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Konfüzyon Matrisi")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.tight_layout()
plt.show()
