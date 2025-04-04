import os
import streamlit as st
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as T
from utils import load_model, predict_mask

os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"
os.environ["STREAMLIT_WATCHED_MODULES"] = ""

# Streamlit app configuration
st.set_page_config(page_title="DentalVision - Teeth Segmentation", layout="centered")
st.title("🦷 DentalVision - Teeth Segmentation App")

# Load the pre-trained model
model = load_model("model/vit_teeth_segmentation.pth")

# Upload interface
uploaded_file = st.file_uploader("Upload a Dental X-ray Image", type=["jpg", "png"])

# Process the uploaded image
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run the segmentation model
    if st.button("Run Segmentation"):
        mask = predict_mask(model, image)
        st.image(mask, caption="Predicted Mask", use_container_width=True)
