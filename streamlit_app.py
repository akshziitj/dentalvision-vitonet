# Import required libraries
import streamlit as st
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as T
from utils import load_model, predict_mask  # Custom utility functions for loading model and predicting mask

# Configure the Streamlit page settings
st.set_page_config(
    page_title="DentalVision - Teeth Segmentation", 
    layout="centered"
)

# Title of the web app
st.title("🦷 DentalVision - Teeth Segmentation App")

# Load the pre-trained segmentation model
model = load_model("model/vit_teeth_segmentation.pth")

# File uploader widget for user to upload dental X-ray images
uploaded_file = st.file_uploader(
    "Upload a Dental X-ray Image", 
    type=["jpg", "png"]
)

# If a file is uploaded, display the image and allow prediction
if uploaded_file:
    # Open the uploaded image and convert it to RGB format
    image = Image.open(uploaded_file).convert("RGB")

    # Display the uploaded image in the app
    st.image(
        image, 
        caption="Uploaded Image", 
        use_container_width=True  # Updated to avoid deprecation warning
    )

    # Button to trigger segmentation
    if st.button("Run Segmentation"):
        # Call the utility function to predict the segmentation mask
        mask = predict_mask(model, image)

        # Display the predicted mask
        st.image(
            mask, 
            caption="Predicted Mask", 
            use_container_width=True  # Updated to avoid deprecation warning
        )
