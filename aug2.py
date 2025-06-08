import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

# Parametreler
image_size = 64  # Görüntü boyutu (64x64)
latent_dim = 128 # Gizli uzay boyutu
batch_size = 32
epochs = 100

# Veri setinizin yolu
data_dir = './dataset' # Lütfen burayı kendi veri setinizin yoluyla değiştirin

# 1. Veri setini yükleme ve ön işleme
def load_and_preprocess_data(data_dir, image_size, batch_size):
    images = []
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = tf.keras.utils.load_img(img_path, target_size=(image_size, image_size))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = (img_array.astype(np.float32) - 127.5) / 127.5  # -1 ile 1 arasında ölçeklendirme
            images.append(img_array)
    images = np.array(images)
    dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(buffer_size=1024).batch(batch_size)
    return dataset

dataset = load_and_preprocess_data(data_dir, image_size, batch_size)

# 2. Üretici (Generator) modelini tanımlama
def build_generator(latent_dim):
    model = models.Sequential()
    model.add(layers.Dense(4*4*256, use_bias=False, input_shape=(latent_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((4, 4, 256)))
    assert model.output_shape == (None, 4, 4, 256)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')) # 3 renk kanalı (RGB)
    assert model.output_shape == (None, 64, 64, 3)

    return model

generator = build_generator(latent_dim)

# 3. Ayrıştırıcı (Discriminator) modelini tanımlama
def build_discriminator():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', input_shape=(image_size, image_size, 3)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = build_discriminator()

# 4. Kayıp fonksiyonlarını ve optimizasyon algoritmalarını tanımlama
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# 5. Eğitim döngüsünü tanımlama
@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, latent_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables())
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables())

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables()))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables()))

# 6. Modeli eğitme
for epoch in range(epochs):
    for image_batch in dataset:
        train_step(image_batch)
    print(f"Epoch {epoch+1}/{epochs} tamamlandı.")

# 7. Üretilen görüntüleri kaydetme (isteğe bağlı)
num_examples_to_generate = 16
noise = tf.random.normal([num_examples_to_generate, latent_dim])
generated_images = generator(noise, training=False)

for i in range(num_examples_to_generate):
    img = tf.keras.utils.array_to_img(generated_images[i] * 127.5 + 127.5, scale=False)
    img.save(f'generated_image_{i}.png')

print("Üretilen görüntüler kaydedildi.")