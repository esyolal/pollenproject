import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
import numpy as np
import math # Yuvarlama işlemleri için

# --- Parametreler ---
# LÜTFEN ORİJİNAL VERİ SETİ DİZİN YOLUNUZU BURAYA YAZIN!
original_data_dir = './t-dataset'

# Artırılmış görüntülerin kaydedileceği çıktı dizini
output_data_dir = './t-datasetaugmented'

# Her polen sınıfı için hedeflenen toplam örnek sayısı (orijinaller dahil)
target_images_per_class = 50

# Görüntülerin kaydedileceği boyut
target_img_height = 224
target_img_width = 224

# Tekrarlanabilirlik için rastgele tohum
random_seed = 42

print(f"Orijinal veri seti: {original_data_dir}")
print(f"Artırılmış görüntüler '{output_data_dir}' dizinine kaydedilecek.")
print(f"Her polen sınıfı için hedeflenen toplam örnek sayısı: {target_images_per_class}")

# --- Çıktı Dizini Hazırlığı ---
if os.path.exists(output_data_dir):
    print(f"'{output_data_dir}' dizini zaten mevcut. İçeriği siliniyor...")
    shutil.rmtree(output_data_dir)
os.makedirs(output_data_dir, exist_ok=True)
print(f"'{output_data_dir}' dizini oluşturuldu/temizlendi.")

# --- Veri Artırma Ayarları ---
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True, # Polen için dikey çevirme uygun olmayabilir, dikkatli kullanın
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# --- Görüntüleri Artır ve Kaydet ---
print("\nVeri artırma işlemi başlatılıyor...")
class_dirs = [d for d in os.listdir(original_data_dir) if os.path.isdir(os.path.join(original_data_dir, d))]
total_original_images_overall = 0
total_augmented_images_overall = 0

for class_name in class_dirs:
    original_class_path = os.path.join(original_data_dir, class_name)
    output_class_path = os.path.join(output_data_dir, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    images_in_class = [f for f in os.listdir(original_class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    current_image_count = len(images_in_class)
    total_original_images_overall += current_image_count

    print(f"\n  Sınıf '{class_name}': Mevcut {current_image_count} orijinal görüntü.")

    # Orijinal görüntüleri yeni dizine kopyala
    for img_name in images_in_class:
        shutil.copy(os.path.join(original_class_path, img_name), os.path.join(output_class_path, img_name))
        total_augmented_images_overall += 1 # Kopyalanan orijinalleri sayıma dahil et

    # Kaç adet artırılmış görüntüye ihtiyacımız var?
    needed_augmented_count = target_images_per_class - current_image_count

    if needed_augmented_count <= 0:
        print(f"    Bu sınıfta zaten {current_image_count} veya daha fazla görüntü var. Artırma yapılmıyor.")
        continue # Bu sınıf için artırma yapmaya gerek yok, sonraki sınıfa geç

    # Her orijinal görüntüden yaklaşık kaç kopya üretmeliyiz?
    # Örneğin, 5 orijinaliniz varsa ve 45 ek görüntüye ihtiyacınız varsa,
    # 45 / 5 = 9 kopya her orijinalden.
    copies_per_original = math.ceil(needed_augmented_count / current_image_count) if current_image_count > 0 else needed_augmented_count

    print(f"    {needed_augmented_count} adet artırılmış görüntüye ihtiyaç var. Her orijinalden yaklaşık {copies_per_original} kopya üretilecek.")
    
    generated_count_this_class = 0

    # Her orijinal görüntüyü işle ve artırılmış kopyaları oluştur
    for img_name in images_in_class:
        img_path = os.path.join(original_class_path, img_name)
        img = tf.keras.utils.load_img(img_path, target_size=(target_img_height, target_img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Batch boyutunu ekle (1, H, W, C)

        i = 0
        for batch in datagen.flow(img_array,
                                  batch_size=1,
                                  save_to_dir=output_class_path,
                                  save_prefix=f'aug_{os.path.splitext(img_name)[0]}',
                                  save_format='jpeg',
                                  seed=random_seed):
            i += 1
            generated_count_this_class += 1
            total_augmented_images_overall += 1 # Üretilen kopyaları genel sayıma dahil et

            # Hedeflenen artırılmış sayıya ulaşıldığında dur
            if generated_count_this_class >= needed_augmented_count:
                break
            # Veya bu orijinalden gereken kopya sayısına ulaşıldığında dur
            if i >= copies_per_original:
                 break
        if generated_count_this_class >= needed_augmented_count:
            break # Sınıf için hedefe ulaşıldığında diğer orijinal görüntüleri işlemeyi bırak

    print(f"    Sınıf '{class_name}' için toplam {current_image_count + generated_count_this_class} görüntü (hedef: {target_images_per_class}).")


print(f"\nVeri artırma işlemi tamamlandı.")
print(f"Genel toplam orijinal görüntü sayısı: {total_original_images_overall}")
print(f"Genel toplam artırılmış görüntü sayısı (orijinaller dahil): {total_augmented_images_overall}")
print(f"Yeni veri seti '{output_data_dir}' dizininde oluşturuldu.")
print(f"Her sınıfın son görüntü sayısını kontrol etmek için '{output_data_dir}' dizinini inceleyebilirsiniz.")

# Opsiyonel: Son görüntü sayılarını doğrula
print("\n--- Artırma Sonrası Sınıf Görüntü Sayıları ---")
for class_name in class_dirs:
    class_path = os.path.join(output_data_dir, class_name)
    if os.path.exists(class_path):
        final_count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"  Sınıf '{class_name}': {final_count} görüntü")
    else:
        print(f"  Sınıf '{class_name}': (Klasör bulunamadı)")