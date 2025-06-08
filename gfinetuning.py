import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB3
import os

# --- 1. Parametreler ---
data_dir = './tdatasetaugmented'
img_height = 300
img_width = 300
batch_size = 32
validation_split_ratio = 0.2
random_seed = 123
model_load_path = 'efficientnetb3_grayscale_model.keras'
model_finetuned_path = 'efficientnetb3_grayscale_finetuned_model.keras'

# --- 2. Dataset Yükleme (Grayscale To RGB) ---
def grayscale_to_rgb(image, label):
    image = tf.image.rgb_to_grayscale(image)  # önce grayscale'e çevir
    image = tf.image.grayscale_to_rgb(image)  # tekrar 3 kanala kopyala
    image = preprocess_input(image)
    return image, label

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

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(grayscale_to_rgb).cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.map(grayscale_to_rgb).cache().prefetch(AUTOTUNE)

# --- 3. Önceden Eğitilmiş Modeli Yükle ---
model = tf.keras.models.load_model(model_load_path)

# --- 4. Fine-Tuning İçin Base Modeli Aç ---
base_model = model.layers[2]  # [0]: input, [1]: aug, [2]: EfficientNet
base_model.trainable = True

# İsteğe bağlı: sadece son 20 katmanı eğit
# for layer in base_model.layers[:-20]:
#     layer.trainable = False

# --- 5. Derleme (düşük learning rate) ---
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- 6. Fine-Tuning Eğitimi ---
fine_tune_epochs = 10
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=fine_tune_epochs
)

# --- 7. Kaydet ---
model.save(model_finetuned_path)
print(f"\n✅ Fine-tuned model başarıyla kaydedildi: {model_finetuned_path}")
