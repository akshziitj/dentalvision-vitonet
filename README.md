# 🦷 DentalVision: Automated Teeth Segmentation with ViT-UNet

[![Streamlit App](https://img.shields.io/badge/Live-Demo-brightgreen?logo=streamlit)](https://dentalvision-vitonet.streamlit.app)
[![Hugging Face Spaces](https://img.shields.io/badge/Hosted-HuggingFace-blue?logo=huggingface)](https://huggingface.co/spaces/akshziitj/dentalvision-vitonet)
[![License](https://img.shields.io/github/license/akshziitj/dentalvision-vitonet)](LICENSE)

DentalVision is a deep learning-based web app that performs **teeth segmentation on dental X-ray images** using a custom-built **ViT-UNet model**. Designed for dentists and researchers to quickly visualize tooth structures.

---

## 🚀 Features

- 🧠 Vision Transformer (ViT) encoder + UNet-style decoder
- 📊 Trained on [Humans in the Loop](https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images) dataset
- ⚡ Simple and fast web UI with Streamlit
- 💾 Upload your own dental X-ray and get segmentation mask
- 📦 One-click deployment on Hugging Face or Streamlit Cloud

---

## 📁 Dataset

We used the **"Teeth Segmentation on Dental X-ray Images"** dataset from Kaggle:

- Source: [Kaggle Dataset](https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images)
- Includes: Original X-rays and annotated masks from human labelers

---

## 🌐 Live Demo

👉 [Try the Streamlit App](https://dentalvision-vitonet.streamlit.app)

👉 [Hugging Face Space](https://huggingface.co/spaces/akshziitj/dentalvision-vitonet)

---

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/akshziitj/dentalvision-vitonet.git
cd dentalvision

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# Install requirements
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app.py
