import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# --- 1. Parametreler ---
data_dir = './tdatasetaugmented'
img_height = 300
img_width = 300
batch_size = 32
validation_split_ratio = 0.2
random_seed = 123
model_load_path = 'efficientnetb3_polen_model.keras'
model_save_path = 'efficientnetb3_polen_model_finetuned.keras'

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

# --- 3. Pipeline ---
from tensorflow.keras.applications.efficientnet import preprocess_input

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y)).cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y)).cache().prefetch(AUTOTUNE)

# --- 4. Modeli Yükle ---
model = tf.keras.models.load_model(model_load_path)

# --- 5. Base modeli eğitime aç ---
base_model = model.layers[2]  # input: 0, data_aug: 1, efficientnet: 2
base_model.trainable = True

# İsteğe bağlı: sadece son 20 katmanı eğit
# for layer in base_model.layers[:-20]:
#     layer.trainable = False

# --- 6. Derleme (küçük learning rate) ---
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- 7. Fine-tuning Eğitimi ---
fine_tune_epochs = 10
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=fine_tune_epochs
)

# --- 8. Kaydet ---
model.save(model_save_path)
print(f"\n✅ Fine-tuned model başarıyla kaydedildi: {model_save_path}")
