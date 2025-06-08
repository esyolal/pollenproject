# 🌿 Pollen Classification with EfficientNetB3

This project focuses on classifying microscopic pollen images using deep learning techniques. An EfficientNetB3 model was fine-tuned to distinguish between different pollen species captured from academic microscopy sources.

---

## 📂 Dataset

Microscopic pollen images were collected and labeled from a scientific textbook and uploaded to Kaggle:

🔗 **Kaggle Dataset**:  
[https://www.kaggle.com/datasets/emresebatiyolal/pollen-image-dataset](https://www.kaggle.com/datasets/emresebatiyolal/pollen-image-dataset)

- Multiple pollen species
- Original high-resolution color microscope images
- Folder structure: one class per folder

---

## 🤖 Model

A CNN model based on **EfficientNetB3** architecture was trained on the dataset.

- Input shape: 300x300 RGB
- Preprocessing: Keras `preprocess_input`
- Trained using grayscale-to-RGB converted images for consistency
- Top-5 class probabilities shown in output

📦 **Download Trained Model (.keras)**:  
[Google Drive – Model Files](https://drive.google.com/drive/folders/1-LZD__zlCWm8BpJfb4EuYjl4GXh71PTv?usp=drive_link)

> Note: Model files are not included in this repo due to size limits.

---

## 🚀 Gradio Interface

You can interact with the model using a simple Gradio web app.

### Run Locally

```bash
pip install -r requirements.txt
python app.py
