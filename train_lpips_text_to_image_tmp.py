#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and


###########################################################################################
#
# Original: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py 
#
#
#############################################################################################

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
# from comet_ml import Experiment
# from comet_ml.integration.pytorch import log_model
import accelerate
# import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
# from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler
###########################
#
# Valerian FOUREL
from models.unet_2d_condition import UNet2DConditionModel
from models.mutual_self_attention import ReferenceAttentionControl
from models.champ_flame_model import ChampFlameModel
from models.guidance_encoder import GuidanceEncoder
from transformers import CLIPVisionModelWithProjection
from datasets.image_dataset import ImageDataset
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from accelerate import DistributedDataParallelKwargs
from lpips import LPIPS  # Assuming you have the LPIPS library available
from visualization.visualization_utils import tensor_to_grid_picture
import torch
from torch.nn.parallel import parallel_apply
from functools import partial
from datetime import datetime

negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

###########################
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel , compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image
from pipeline.pipeline_stable_diffusion import StableDiffusionPipeline

import subprocess
import sys

# # Use this to install wandb
# # Example usage to install `wandb`


import wandb
wandb.init(project="ChampFace")
import json


#annotation_file = '/home/vfourel/FaceGPT/Data/LLaVAAnnotations/StableDiffusionPrompts/prompt_response_conversation_All_data.json'
model_id = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
annotation_file = '/home/vfourel/FaceGPT/Data/LLaVAAnnotations/StableDiffusionPrompts/PromptsSmall07_41ksamples.json'

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
    
        # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.22.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


def save_model_card(
    args,
    repo_id: str,
    images=None,
    repo_folder=None,
):
    img_str = ""
    if len(images) > 0:
        image_grid = make_image_grid(images, 1, len(args.validation_prompts))
        image_grid.save(os.path.join(repo_folder, "val_imgs_grid.png"))
        img_str += "![val_imgs_grid](./val_imgs_grid.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {args.pretrained_model_name_or_path}
datasets:
- {args.dataset_name}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
inference: true
---
    """
    model_card = f"""
# Text-to-image finetuning - {repo_id}

This pipeline was finetuned from **{args.pretrained_model_name_or_path}** on the **{args.dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompts: {args.validation_prompts}: \n
{img_str}

## Pipeline usage

You can use the pipeline like so:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("{repo_id}", torch_dtype=torch.float16)
prompt = "{args.validation_prompts[0]}"
image = pipeline(prompt).images[0]
image.save("my_image.png")
```

## Training info

These are the key hyperparameters used during training:

* Epochs: {args.num_train_epochs}
* Learning rate: {args.learning_rate}
* Batch size: {args.train_batch_size}
* Gradient accumulation steps: {args.gradient_accumulation_steps}
* Image resolution: {args.resolution}
* Mixed-precision: {args.mixed_precision}

"""
    wandb_info = ""
    if is_wandb_available():
        wandb_run_url = None
        if wandb.run is not None:
            wandb_run_url = wandb.run.url

    if wandb_run_url is not None:
        wandb_info = f"""
More information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb_run_url}).
"""

    model_card += wandb_info

    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype,guidance_encoder_flame,save_picture=False):
    logger.info("Running validation... ")

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        # guidance_encoder_flame = guidance_encoder_flame, # we add the guidance encoder
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

  # Load prompts from the demonstration file
    with open(args.demonstration_file, 'r') as f:
        prompts_dict = json.load(f)

    images = []
    for key, prompt in prompts_dict.items():
        print(f"Generating image for prompt: {prompt}")
        with torch.autocast("cuda"):
            image = pipeline(prompt=prompt,
             negative_prompt=negative_prompt,
              num_inference_steps=900,
               generator=generator,
               height=args.image_height,  # Set the desired height
                width=args.image_width     # Set the desired width
                ).images[0]
        images.append(image)

    # Convert images to a tensor
    image_tensors = torch.stack([torch.from_numpy(np.array(img)).permute(2, 0, 1) for img in images])

    # Create the output directory if it doesn't exist
    output_dir = args.output_sample_image
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate grid picture
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid_filename = f"grid_picture_{timestamp}.png"
    grid_path = os.path.join(output_dir, grid_filename)
    tensor_to_grid_picture(image_tensors, output_dir, filename=grid_filename)

    print(f"Grid image saved at {grid_path}")
        # Save individual images
    if save_picture:
        for i, image in enumerate(images):
            image_filename = f"image_{i}_{timestamp}.png"
            image_path = os.path.join(output_dir, image_filename)
            image.save(image_path)
            print(f"Individual image saved at {image_path}")

    del pipeline
    torch.cuda.empty_cache()

    return images

########################################################################
# VF: function to get the predicted image from the noise residuals
# We want to predict the image from the noise residuals and send it back to 
# LPIPs to compare with originals and get a loss
# 
def process_batch_image_pred_old(model_pred, timesteps, noisy_latents, noise_scheduler, vae):
    batch_size = model_pred.shape[0]
    denoised_latents_list = []

    for i in range(batch_size):
        denoised_latents = noise_scheduler.step(
            model_output=model_pred[i],
            timestep=timesteps[i],
            sample=noisy_latents[i]
        ).pred_original_sample

        denoised_latents = 1 / 0.18215 * denoised_latents
        denoised_latents_list.append(denoised_latents)


    denoised_latents_batch = torch.stack(denoised_latents_list)
    #print(denoised_latents_batch.shape, denoised_latents_batch[0].shape)

    image_pred = vae.decode(denoised_latents_batch.unsqueeze(0)).sample

    return image_pred



def process_single_image(model_pred, timestep, noisy_latent, noise_scheduler,vae,model_dtype):
    model_pred = model_pred.to(dtype=model_dtype)
    noisy_latent = noisy_latent.to(dtype=model_dtype)
    denoised_latents = noise_scheduler.step(
        model_output=model_pred,
        timestep=timestep,
        sample=noisy_latent
    ).pred_original_sample
    scaled_denoised_latents = 1 / 0.18215 * denoised_latents
    # print(scaled_denoised_latents.shape)
    image_pred =  vae.decode(scaled_denoised_latents.unsqueeze(0)).sample
    return image_pred

def process_batch_image_pred(model_pred, timesteps, noisy_latents, noise_scheduler, vae,model_dtype):
    batch_size = model_pred.shape[0]
    

    # Create a partial function with fixed noise_scheduler
    process_fn = partial(process_single_image, noise_scheduler=noise_scheduler,vae=vae,model_dtype = model_dtype )

    # Prepare inputs for parallel processing
    inputs = list(zip(model_pred, timesteps, noisy_latents))

    # Process in parallel
    denoised_latents_list = parallel_apply(
        [process_fn] * batch_size,
        inputs
    )
    # denoised_latents_list = []

    # for i in range(batch_size):
    #     denoised_latent = process_single_image(
    #         model_pred[i],
    #         timesteps[i],
    #         noisy_latents[i],
    #         noise_scheduler,
    #         vae
    #     )
    #     denoised_latents_list.append(denoised_latent)

    # Stack the results
    denoised_latents_batch = torch.stack(denoised_latents_list).squeeze(1)

    # Decode the entire batch at once
    image_pred = denoised_latents_batch #vae.decode(denoised_latents_batch).sample
    # print(denoised_latents_batch.shape, denoised_latents_batch[0].shape)

    return image_pred

def normalize_between_neg1_and_1(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized = 2 * (tensor - min_val) / (max_val - min_val) - 1
    return normalized
##################################################################


def main(args):

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    
    #########################################################
    # VF
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)


    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs]
    )
    #########################################################

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
    #    datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
       #s.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        )

    ####################################################################################
    # From VF VF
    # modified 
    #

    reference_unet = UNet2DConditionModel.from_pretrained(
        args.base_model_path,
        subfolder="unet",
    ).to(device="cuda")


    # image_enc = CLIPVisionModelWithProjection.from_pretrained(
    #     args.image_encoder_path,
    # ).to(dtype=weight_dtype, device="cuda")    
    
    guidance_encoder_flame = setup_guidance_encoder(args)
    
    # Freeze some modules
    vae.requires_grad_(False)
    ########################################
    # VF: WE NEED THIS for the lpips loss 
    #
    model_dtype = next(vae.parameters()).dtype
    ########################################
    #image_enc.requires_grad_(False)
    for name, param in reference_unet.named_parameters():
        if "up_blocks.3" in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)
            
    for module in guidance_encoder_flame.values():
        module.requires_grad_(True)
            
    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )

  #############################################################
  # modified by VF
  #
    model = ChampFlameModel(
            reference_unet,
            reference_control_writer,
            guidance_encoder_flame,
        )
    
    # Initialize LPIPS loss
    lpips_loss = LPIPS(net='vgg').to(accelerator.device)  # You can use 'alex', 'vgg', or 'squeeze' as the network
###################################################################################
    if args.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()        
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )
        
    if args.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
 # look above to generate the code for the models VF VF
####################################################################################


#### Take this into Account 
# set to true
    # Freeze vae and text_encoder and set unet to trainable
    # this is the original
    vae.requires_grad_(False)


    text_encoder.requires_grad_(False)
    model.train() # VF: modified from unet





    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
        )
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            reference_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            print("\n\n\n\n In save model_hook \n\n\n\n")
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet", added_cond_kwargs={})
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
###################################################
# VF : modifed from Champ
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))

    optimizer = optimizer_cls(
        trainable_params, # VF
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
###################################################


    # # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )






#######################################################
#
# Modifications by VF
#
    train_dataset = ImageDataset(
        tokenizer = tokenizer,
        text_encoder=text_encoder,
        image_json_path=args.data.image_json_path,
        image_size=args.data.image_size,
        sample_margin=args.data.sample_margin,
        data_parts=args.data.data_parts,
        guids=args.data.guids,
        extra_region=None,
        bbox_crop=args.data.bbox_crop,
        bbox_resize_ratio=tuple(args.data.bbox_resize_ratio),
    )


######################################################################
#
# VF: modify this function


    def collate_fn(examples):
        # Assuming `tgt_img` contains the image data
        pixel_values = torch.stack([example["tgt_img"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        
        # Assuming `tgt_guid` contains tensor data
        tgt_guids = torch.stack([example["tgt_guid"] for example in examples])
        attention_masks = torch.stack([example["attention_mask"] for example in examples])
        input_ids = torch.stack([example["input_ids"] for example in examples])
        # print("DEVICE:    ",input_idss.device,attention_masks.device)
        # with torch.no_grad():  # If you're not training the text encoder, use no_grad to save memory
        #      text_embeddings = text_encoder(input_idss, attention_mask=attention_masks)[0]

        # If `description` is a list of strings, you may need to tokenize them
        # For this example, let's assume descriptions are already tokenized
            # Extract and tokenize descriptions
        # text_embeddings = [example["text_embeddings"] for example in examples]

        return {
            "pixel_values": pixel_values,
            "tgt_guids": tgt_guids,
            "attention_masks":attention_masks,
            "input_ids":input_ids
        }

    ######################################################################

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True # for smoother training
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    # VF: we modified it from unet to model
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    ##################################
    #
    # We remove the last batch to keep a smooth training function
    #

    # # Determine the total number of batches
    # total_batches = len(train_dataloader)

    # # Create a new dataloader without the last batch
    # train_dataloader = itertools.islice(train_dataloader, total_batches - 1)

    # # If needed, wrap the iterator back to a DataLoader
    # train_dataloader = DataLoader(list(train_dataloader), batch_size=train_dataloader.batch_size, shuffle=train_dataloader.shuffle)




    #######################################################
    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    ############################################################################################################
    #
    # Modifications by VF
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        print(tracker_config)
        tracker_config.pop("validation_prompts", None)
    # we need to define the model_dtype
    model_dtype = next(vae.parameters()).dtype
        # Initialize only WandB tracker, skip TensorBoard
        # accelerator.init_trackers(args.tracker_project_name, config=tracker_config, init_kwargs={"wandb": {"entity": "your_wandb_entity"}})
        ##############################################################################################################################
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    wandb.config.update({
    "learning_rate": args.learning_rate,
    "batch_size": args.train_batch_size,
    "train_epochs": args.num_train_epochs,
    "model_name": args.pretrained_model_name_or_path,
    "resolution": args.resolution
})
    counter = 0
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        # Determine the total number of batches
        total_batches = len(train_dataloader)
        # We remove the last batch to keep a smooth training function
        # Calculate half of the batches
        quarter_batches = num_update_steps_per_epoch // 4
        half_batches = num_update_steps_per_epoch //2
        for step, batch in enumerate(train_dataloader):

            if step == num_update_steps_per_epoch - 1:
                continue
            # Calculate if we're at half of the batches or the last batch
            
            is_quarter_or_halfway_or_last = (counter % (quarter_batches*args.gradient_accumulation_steps) == 0) # (step % (quarter_batches*3 - 1) == 0 ) or (step % (quarter_batches - 1) == 0 ) or (step % (half_batches - 1) == 0) or
            #print('counter % (quarter_batches)', counter % (quarter_batches))
            if counter == 0:
                is_quarter_or_halfway_or_last = False
            with torch.no_grad():  # If you're not training the text encoder, use no_grad to save memory
                  text_embeddings = text_encoder(batch["input_ids"])[0]

            counter += 1
            ###### figure out the training using guidance
            with accelerator.accumulate(model):
                # Convert images to latent space
                # you want to compare the batch["pixel_values"], with the noise_repdiction 
                # passed by the vae (which is what reconstruct the images)
                # you then want to do  lpips to it.
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                # Prepare additional conditional kwargs
                # Assuming time_ids are part of your batch or need to be set up previously
                batch_size, seq_length, feature_size = encoder_hidden_states.shape

                time_ids = [0] * encoder_hidden_states.shape[0]  # Adjust .shape[0] to .shape[1] if seq_length is desired
                # expanded to have 1 as a placeholder for feature size
                time_ids = torch.zeros(encoder_hidden_states.shape, dtype=torch.long, device=encoder_hidden_states.device)
                time_ids = torch.zeros(batch_size, dtype=torch.int, device=encoder_hidden_states.device)
                # Get the shape of encoder_hidden_states


                # Example of a projection layer if you want to map the zeros to a meaningful feature space
                # Here we keep it simple by just repeating the zero across the feature dimension

                # Get the target for loss depending on the prediction type
                added_cond_kwargs = {}
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                

                #################################################################################################################
                # Modified by VF
                #


                # Predict the noise residual and compute loss
                model_pred = model(noisy_latents, timesteps,encoder_hidden_states= encoder_hidden_states,multi_guidance_cond = batch["tgt_guids"])
                model_pred = model_pred.sample
                #     # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                #     # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                #     # This is discussed in Section 4.2 of the same paper.

                if args.snr_gamma == 0:
                    l1_loss = F.l1_loss(model_pred.float(), target.float(), reduction="mean")
                    if args.lpips_loss_weight == 0:
                        lpips_value = 0.0
                    else:
                        image_pred_batch = process_batch_image_pred(model_pred, timesteps, noisy_latents, noise_scheduler, vae,model_dtype)
                        lpips_value = lpips_loss(normalize_between_neg1_and_1(image_pred_batch.float()), normalize_between_neg1_and_1(batch["pixel_values"].to(weight_dtype).float())).mean()
                    loss = args.l1_loss_weight * l1_loss + args.lpips_loss_weight * lpips_value

                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    l1_loss_weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )

                    l1_loss = F.l1_loss(model_pred.float(), target.float(), reduction="none")
                    l1_loss = l1_loss.mean(dim=list(range(1, len(l1_loss.shape)))) * l1_loss_weights
                    l1_loss = l1_loss.mean()
                        # perform guidance
                    #print(model_pred.shape)
                    # print('image_pred_batch',image_pred_batch.shape)
                    # print('timesteps',timesteps)
                    # print('noisy_latents',noisy_latents.shape)
                    #tensor_to_grid_picture(image_pred_batch,'image_check','pred.png')
                    #tensor_to_grid_picture(batch["pixel_values"].to(weight_dtype),'image_check','original.png')
                     # Calculate LPIPS loss
                     # The exact method depends on your noise scheduler. Here's a general approach:
                    # denoised_latents = noise_scheduler.step(
                    #      model_output=model_pred[0],
                    #     timestep=timesteps[0],
                    #      sample=noisy_latents[0]
                    #  ).pred_original_sample
                    #print(denoised_latents)
                    # denoised_latents = 1 / 0.18215 * denoised_latents
                    # try:
                    #     scaling_factor = reference_unet.config.scaling_factor
                    #     print(scaling_factor, 'scaling_factor')
                    # except:
                    #     print('error')

                    # print(denoised_latents.shape,denoised_latents[0].shape)
                    #image_pred = vae.decode(denoised_latents.unsqueeze(0)).sample
                    # print("VAE config:", vae.config)
                    
                    #print("image_pred shape:", image_pred.shape, model_pred.shape)
                    # print("original shape:", batch["pixel_values"].to(weight_dtype).shape)
                    if args.lpips_loss_weight == 0:
                        lpips_value = 0.0
                    else:
                        image_pred_batch = process_batch_image_pred(model_pred, timesteps, noisy_latents, noise_scheduler, vae,model_dtype)

                        lpips_value = lpips_loss(normalize_between_neg1_and_1(image_pred_batch.float()), normalize_between_neg1_and_1(batch["pixel_values"].to(weight_dtype).float())).mean()
                    
                    loss = args.l1_loss_weight * l1_loss + args.lpips_loss_weight * lpips_value



                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                # loss.backward()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(reference_unet.parameters(), args.max_grad_norm)
                    # accelerator.clip_grad_norm_(vae.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                ############################################################################
                # VF: unallocated resources for the GPUs
                #
                if args.lpips_loss_weight == 0:
                    torch.cuda.empty_cache()
                else:
                    torch.cuda.empty_cache()
                    del image_pred_batch
                    del lpips_value
                    del l1_loss
                ############################################################################

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                # no wandb by VF
                # accelerator.log({"train_loss": train_loss,"loss": loss, "learning_rate": lr_scheduler.get_last_lr()}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        if torch.cuda.device_count() == 1:
                            model.save_model(save_path)  # Save model directly if only one GPU is available
                        else:
                            accelerator.save_state(save_path)  # Save accelerator state if multiple GPUs are used
                        logger.info(f"Saved state to {save_path}")

            # Log to wandb
            wandb.log({
                "train_loss": train_loss,
                "loss": loss.item(),
                "learning_rate": lr_scheduler.get_last_lr()[0]
            }, step=global_step)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            logs = {"step_loss": loss.detach().item(), "learning_rate": lr_scheduler.get_last_lr()[0]}

            if global_step >= args.max_train_steps:
                break
            if is_quarter_or_halfway_or_last: # we run the validation evey half epochs
            # Ensure all processes are synchronized before checking if it's the main process
                #accelerator.wait_for_everyone()
                print('Running the validation script')
                accelerator.wait_for_everyone()
                log_validation(
                    vae,
                    text_encoder,
                    tokenizer,
                    reference_unet,
                    args,
                    accelerator,
                    weight_dtype,
                    guidance_encoder_flame
                )


        if accelerator.is_main_process:
            # Run validation every half epoch
            if is_quarter_or_halfway_or_last: # we run the validation evey half epochs
            # Ensure all processes are synchronized before checking if it's the main process
                #accelerator.wait_for_everyone()
                print('Running the validation script')
                accelerator.wait_for_everyone()
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(reference_unet.parameters())
                    ema_unet.copy_to(reference_unet.parameters())
                #####################################
                #
                # Valerian FOUREL
                # guidance_encoder_flame = setup_guidance_encoder(args)
                # for module in guidance_encoder_flame.values():
                #     module.requires_grad_(True)
                #####################################
                log_validation(
                    vae,
                    text_encoder,
                    tokenizer,
                    reference_unet,
                    args,
                    accelerator,
                    weight_dtype,
                    guidance_encoder_flame
                )


                if args.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(reference_unet.parameters())

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(reference_unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder, # we have to 
            vae=vae,
            unet=reference_unet,
            guidance_encoder_flame=guidance_encoder_flame,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)

        # Run a final round of inference.
        log_validation(
                    vae,
                    text_encoder,
                    tokenizer,
                    reference_unet,
                    args,
                    accelerator,
                    weight_dtype,
                    guidance_encoder_flame
                )

        if args.push_to_hub:
            save_model_card(args, repo_id, images, repo_folder=args.output_dir)
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()

#######################################
#
# Modification by Valerian Fourel
#

def setup_guidance_encoder(cfg):
    guidance_encoder_group = dict()

    for guidance_type in cfg.data.guids:
        guidance_encoder_group[guidance_type] = GuidanceEncoder(
            guidance_embedding_channels=cfg.guidance_encoder_kwargs.guidance_embedding_channels,
            guidance_input_channels=cfg.guidance_encoder_kwargs.guidance_input_channels,
            block_out_channels=cfg.guidance_encoder_kwargs.block_out_channels,
        )

    return guidance_encoder_group

#######################################
if __name__ == "__main__":
    #######################################
    # VF
    import shutil
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train/flame_train_lpips.yaml")
    args = parser.parse_args()

    if args.config.endswith(".yaml"):
        config = OmegaConf.load(args.config)
    else:
        raise ValueError("Do not support this format config file")

    main(config)
#######################################