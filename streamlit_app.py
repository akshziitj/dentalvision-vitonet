import sys
import streamlit as st
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as T
from utils import load_model, predict_mask

# --- Patch to avoid RuntimeError caused by torch.classes introspection ---
if "torch.classes" in sys.modules:
    del sys.modules["torch.classes"]
# -------------------------------------------------------------------------

# Set Streamlit page configuration
st.set_page_config(page_title="DentalVision - Teeth Segmentation", layout="centered")

# App title
st.title("🦷 DentalVision - Teeth Segmentation App")

# Load the trained segmentation model
model = load_model("model/vit_teeth_segmentation.pth")

# Upload interface for dental X-ray images
uploaded_file = st.file_uploader("Upload a Dental X-ray Image", type=["jpg", "png"])

# If an image is uploaded
if uploaded_file:
    # Open the image using PIL and ensure it's in RGB mode
    image = Image.open(uploaded_file).convert("RGB")

    # Display the uploaded image in the app
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Button to trigger segmentation
    if st.button("Run Segmentation"):
        # Predict the mask using the model
        mask = predict_mask(model, image)

        # Show the predicted segmentation mask
        st.image(mask, caption="Predicted Mask", use_container_width=True)
