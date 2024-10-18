
import json
import random
from typing import List
from pathlib import Path

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset

# Define your dataset class
class CustomDataset(Dataset):
    def __init__(self, annotation_file, tokenizer, transform=None):
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        self.transform = transform
        self.image_paths = list(self.annotations.keys())#[:1000]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        description = self.annotations[image_path]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        inputs = self.tokenizer(
            description, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return {"image": image, "input_ids": inputs.input_ids}
    