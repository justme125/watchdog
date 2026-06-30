import cv2
import numpy as np
import open_clip
import torch
from PIL import Image

from config import CLIP_MODEL_NAME, CLIP_PRETRAINED


def load_embedder():
    """Load OpenCLIP and return its model and image transform."""

    model, _, transform = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED,
    )
    model.eval()
    return model, transform


def embed_crop(model, transform, crop_image):
    """Create a normalized, one-dimensional float32 embedding."""

    rgb_crop = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_crop)
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        embedding = model.encode_image(tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    return np.asarray(embedding.squeeze(0).cpu(), dtype=np.float32).reshape(-1)
