import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
from transformers import ViTModel, ViTConfig
import torch.nn.functional as F

class ViTUNet(nn.Module):
    def __init__(self):
        super(ViTUNet, self).__init__()
        config = ViTConfig(image_size=224, patch_size=16, num_channels=3, hidden_size=768,
                           num_attention_heads=12, num_hidden_layers=6)
        self.encoder = ViTModel(config)

        self.decoder = nn.Sequential(
            nn.Conv2d(768, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(pixel_values=x).last_hidden_state
        b, n, c = x.shape
        x = x[:, 1:, :].permute(0, 2, 1).reshape(b, c, 14, 14)
        x = self.decoder(x)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return x

def load_model(path):
    model = ViTUNet()
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model

def predict_mask(model, image):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)[0][0]
    mask = (output > 0.5).float().numpy()
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    return mask_img
