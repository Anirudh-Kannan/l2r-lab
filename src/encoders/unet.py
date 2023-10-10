import cv2
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import torchvision.transforms as transforms

from src.config.yamlize import yamlize
from src.constants import DEVICE
from src.encoders.base import BaseEncoder
from src.encoders.transforms.preprocessing import crop_resize_center
from src.encoders.unet.unet_model import UNet


@yamlize
class Unet(BaseEncoder, torch.nn.Module):
    """Input should be (bsz, C, H, W)"""

    def __init__(
        self,
        load_checkpoint_from: str = "",
    ):
        super().__init__()
        self.model = UNet(channel_depth=32, n_channels=3, n_classes=1)

        if load_checkpoint_from == "":
            logging.info("Not loading any visual encoder checkpoint")
        else:
            self.load_state_dict(torch.load(load_checkpoint_from))

        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def encode(self, img: np.ndarray, device=DEVICE) -> torch.Tensor:
        # assume x is RGB image with shape (H, W, 3)
        img = self.tf(img)
        img = img.float()
        v = self.model.encode(img)
        return v

    def decode(self, z):
        pass

    def update(self, batch_of_images):
        pass
