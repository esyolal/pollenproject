import os

# Klasör yolu
dataset_path = './tdatasetaugmented'

# Sadece klasörleri al
class_names = sorted([
    name for name in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, name))
])

# Listeyi yazdır
print("class_names = [")
for name in class_names:
    print(f'    "{name}",')
print("]")
