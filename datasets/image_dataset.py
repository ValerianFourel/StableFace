###########################################
# Valerian Fourel
# The full file needs to be changed given our data
# the original cnabe found at: https://github.com/fudan-generative-vision/champ/blob/master/datasets/image_dataset.py
#

import json
import random
from typing import List
from pathlib import Path

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
from tqdm import tqdm
from datasets.data_utils import process_bbox, crop_bbox, mask_to_bbox, mask_to_bkgd


class ImageDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        text_encoder,
        image_json_path: str,
        image_size: int = 768,
        sample_margin: int = 30,
        data_parts: list = ["all"],
        guids: list = ["flame"], # modified by VF from original
        extra_region: list = [],
        bbox_crop=True,
        bbox_resize_ratio=(0.8, 1.2),
        aug_type: str = "Resize",  # "Resize" or "Padding"
        select_face=False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.image_json_path = image_json_path
        self.image_size = image_size
        self.sample_margin = sample_margin
        self.data_parts = data_parts
        self.guids = guids
        self.extra_region = extra_region
        self.bbox_crop = bbox_crop
        self.bbox_resize_ratio = bbox_resize_ratio
        self.aug_type = aug_type
        self.select_face = select_face
        # data is the dict of the paths and the desriptions of the images
        self.data_lst , self.data = self.generate_data_lst()
        
        self.clip_image_processor = CLIPImageProcessor()
        self.pixel_transform, self.guid_transform = self.setup_transform()
            
    def generate_data_lst(self):
        data = {}
        with open(self.image_json_path, 'r') as file:
            data = json.load(file)
        
        # Extract keys (image paths) from the JSON data
        data_lst = list(data.keys())
        return data_lst, data
    
    def is_valid(self, video_dir: Path):
        video_length = len(list((video_dir / "images").iterdir()))
        for guid in self.guids:
            guid_length = len(list((video_dir / guid).iterdir()))
            if guid_length == 0 or guid_length != video_length:
                return False
        if self.select_face:
            if not (video_dir / "face_images").is_dir():
                return False
            else:
                face_img_length = len(list((video_dir / "face_images").iterdir()))
                if face_img_length == 0:
                    return False
        return True
    
    def resize_long_edge(self, img):
        img_W, img_H = img.size
        long_edge = max(img_W, img_H)
        scale = self.image_size / long_edge
        new_W, new_H = int(img_W * scale), int(img_H * scale)
        
        img = F.resize(img, (new_H, new_W))
        return img

    def padding_short_edge(self, img):
        img_W, img_H = img.size
        width, height = self.image_size, self.image_size
        padding_left = (width - img_W) // 2
        padding_right = width - img_W - padding_left
        padding_top = (height - img_H) // 2
        padding_bottom = height - img_H - padding_top
        
        img = F.pad(img, (padding_left, padding_top, padding_right, padding_bottom), 0, "constant")
        return img
    
    def setup_transform(self):
        if self.bbox_crop:
            if self.aug_type == "Resize":
                pixel_transform = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                guid_transform = transforms.Compose = ([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                ])
                
            elif self.aug_type == "Padding":
                pixel_transform = transforms.Compose([
                    transforms.Lambda(self.resize_long_edge),
                    transforms.Lambda(self.padding_short_edge),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                guid_transform = transforms.Compose([
                    transforms.Lambda(self.resize_long_edge),
                    transforms.Lambda(self.padding_short_edge),
                    transforms.ToTensor(),
                ])
            else:
                raise NotImplementedError("Do not support this augmentation")
        
        else:
            pixel_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=self.image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            guid_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=self.image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
            ])
        
        return pixel_transform, guid_transform            
            
    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.cat(transformed_images, dim=0)  # (c*n, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor
    
    
    def __len__(self):
        return len(self.data_lst)
    
    ### 
    # We simply the get item function in order to only use the 
    # flame guidance
    def __getitem__(self, idx):
        item_path = self.data_lst[idx]
        ####################################
        # to obtain the paths of the flame guidance
        modified_path = item_path.replace("inputs", "geometry_detail")

        

        original_img_pil = Image.open(item_path)
        flame_img_pil = Image.open(modified_path)
        description = self.data.get(item_path)
        text_input = self.tokenizer(description, padding="max_length", truncation=True, return_tensors="pt")

        # Generate text embeddings
        input_ids = text_input["input_ids"]
        attention_mask = text_input["attention_mask"]


        # # Forward pass through text encoder

        # augmentation
        state = torch.get_rng_state()
        tgt_img = self.augmentation(original_img_pil, self.pixel_transform, state)
        tgt_guid = self.augmentation(flame_img_pil, self.guid_transform, state)

        sample = dict(
            tgt_img=tgt_img,
            tgt_guid=tgt_guid,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        return sample