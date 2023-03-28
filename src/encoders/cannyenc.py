import cv2
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from src.config.yamlize import yamlize
from src.constants import DEVICE
from src.encoders.base import BaseEncoder
from src.encoders.transforms.preprocessing import crop_resize_center

@yamlize
class CannyEncoder(BaseEncoder, torch.nn.Module):
    """Input should be (bsz, C, H, W) where C=3, H=42, W=144"""

    def __init__(
        self,
        image_channels: int = 3,
        image_height: int = 42,
        image_width: int = 144,
        z_dim: int = 32,
        load_checkpoint_from: str = "",    
    ):
        super().__init__()

        self.im_c = image_channels
        self.im_h = image_height
        self.im_w = image_width

        input_sz = image_channels * image_height * image_width
        self.z = z_dim
        self.encoder  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_sz, z_dim)
        )

    def encode_canny(self, x, z):
       # Apply Canny edge detection
        edges = cv2.Canny(x, 100, 200)

        # Downsample to a vector of size z using mean pooling
        downsampled_edges = cv2.resize(edges, (z, 1), interpolation=cv2.INTER_AREA)

        # Convert to a PyTorch tensor
        downsampled_edges_torch = torch.from_numpy(downsampled_edges).float() 
        return downsampled_edges_torch

    def encode(self, x: np.ndarray, device = DEVICE) -> torch.Tensor:
        print(x.shape)
        return self.encode_canny(x, self.z)

    def decode(self, z):
        pass

    def update(self, batch_of_images):
        pass

