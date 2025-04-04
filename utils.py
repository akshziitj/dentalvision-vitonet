# Import necessary libraries
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
from transformers import ViTModel, ViTConfig
import torch.nn.functional as F

# Define a custom neural network that combines Vision Transformer (ViT) and U-Net-style decoder
class ViTUNet(nn.Module):
    def __init__(self):
        super(ViTUNet, self).__init__()

        # Configure the ViT encoder
        config = ViTConfig(
            image_size=224,           # Input image size
            patch_size=16,            # Size of image patches
            num_channels=3,           # RGB channels
            hidden_size=768,          # Embedding size
            num_attention_heads=12,   # Number of attention heads
            num_hidden_layers=6       # Number of transformer blocks
        )

        # Initialize the Vision Transformer encoder
        self.encoder = ViTModel(config)

        # Define a decoder that gradually upsamples feature maps to output a 2D mask
        self.decoder = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # Upsample
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),    # Upsample again
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),                        # Final 1-channel output
            nn.Sigmoid()                                            # Normalize output to [0, 1]
        )

    def forward(self, x):
        # Pass input image through the ViT encoder
        x = self.encoder(pixel_values=x).last_hidden_state  # Output: (batch_size, num_patches+1, hidden_size)

        # Extract shape parameters
        b, n, c = x.shape

        # Remove the class token and reshape to 2D feature map
        x = x[:, 1:, :].permute(0, 2, 1).reshape(b, c, 14, 14)  # Convert to (batch, channels, height, width)

        # Pass through the decoder to get the predicted segmentation mask
        x = self.decoder(x)

        # Upsample the output to match original input size (224x224)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        return x


# Function to load the trained model from saved weights
def load_model(path):
    model = ViTUNet()  # Initialize model
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))  # Load weights
    model.eval()  # Set to evaluation mode
    return model

# Function to predict the segmentation mask given a PIL image and model
def predict_mask(model, image):
    # Define preprocessing: resize and convert image to tensor
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    
    # Apply transform and add batch dimension
    image_tensor = transform(image).unsqueeze(0)

    # Disable gradient tracking for inference
    with torch.no_grad():
        output = model(image_tensor)[0][0]  # Get first image in batch and squeeze output

    # Apply threshold to convert probabilities to binary mask
    mask = (output > 0.5).float().numpy()

    # Convert mask to PIL Image format for display
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))

    return mask_img
