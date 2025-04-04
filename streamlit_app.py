import os

# MUST come before any streamlit imports
os.environ["STREAMLIT_WATCH_DISABLE"] = "true"

import streamlit as st
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as T
from utils import load_model, predict_mask

# App config
st.set_page_config(page_title="DentalVision - Teeth Segmentation", layout="centered")
st.title("🦷 DentalVision - Teeth Segmentation App")

# Load model
model = load_model("model/vit_teeth_segmentation.pth")

# Upload image
uploaded_file = st.file_uploader("Upload a Dental X-ray Image", type=["jpg", "png"])

# Display and predict
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Segmentation"):
        mask = predict_mask(model, image)
        st.image(mask, caption="Predicted Mask", use_container_width=True)
