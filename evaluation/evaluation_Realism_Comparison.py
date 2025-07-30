#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate images with a baseline Stable-Diffusion model and a finetuned model,
then compare both sets with the ground-truth reference images using

    • Fréchet Inception Distance (FID, lower = better)

A concise summary is printed and a CSV with the prompts processed is saved.

---------------------------------------------------------------------------
Dependencies  (install with pip)

pip install diffusers transformers omegaconf pandas pillow torchvision \
            pytorch-fid
"""

# --------------------------------------------------------------------- #
#                                Imports                                #
# --------------------------------------------------------------------- #
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image

# Stable-Diffusion ------------------------------------------------------
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers import StableDiffusionPipeline as SDPipelineOriginal
from transformers import CLIPTokenizer, CLIPTextModel

# Metric ----------------------------------------------------------------
from pytorch_fid import fid_score

# Project-specific modules ---------------------------------------------
from models.champ_flame_model import ChampFlameModel
from models.guidance_encoder import GuidanceEncoder
from models.mutual_self_attention import ReferenceAttentionControl
from models.unet_2d_condition import UNet2DConditionModel
from pipeline.pipeline_stable_diffusion import StableDiffusionPipeline

# --------------------------------------------------------------------- #
#                              Constants                                #
# --------------------------------------------------------------------- #
NEG_PROMPT_DEFAULT = (
    "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, "
    "cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, "
    "worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, "
    "mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn "
    "face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions,"
    " extra limbs, cloned face, disfigured, gross proportions, malformed limbs,"
    " missing arms, missing legs, extra arms, extra legs, fused fingers, too "
    "many fingers, long neck"
)

# --------------------------------------------------------------------- #
#                       Stable-Diffusion helpers                        #
# --------------------------------------------------------------------- #
def load_guidance_encoder(cfg):
    # Define the path to the pre-trained weights
    guidance_encoder_group = dict()
    # Move the model to the appropriate device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for guidance_type in cfg.data.guids:
        guidance_encoder_group[guidance_type] = GuidanceEncoder(guidance_embedding_channels=cfg.guidance_encoder_kwargs.guidance_embedding_channels,
            guidance_input_channels=cfg.guidance_encoder_kwargs.guidance_input_channels,
            block_out_channels=cfg.guidance_encoder_kwargs.block_out_channels,)
        state_dict_guidance_encoder = torch.load(cfg.pretrained_path_guidance_encoder[guidance_type], map_location=torch.device('cpu'))
        # Check if the loaded state_dict is wrapped (e.g., with DataParallel)
        if "module." in list(state_dict_guidance_encoder.keys())[0]:
            # Remove the "module." prefix
            state_dict_guidance_encoder = {k.replace("module.", ""): v for k, v in state_dict_guidance_encoder.items()}

        guidance_encoder_group[guidance_type].load_state_dict(state_dict_guidance_encoder)
        guidance_encoder_group[guidance_type].to(device)
        guidance_encoder_group[guidance_type].eval()

    return guidance_encoder_group

def load_models(cfg):
    # Load tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="text_encoder")

    # Load VAE
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="vae")

    # Load UNet
    reference_unet = UNet2DConditionModel.from_pretrained(cfg.finetuned_model, subfolder="unet")

    # Setup guidance encoder
    guidance_encoders = load_guidance_encoder(cfg)

    # Create ReferenceAttentionControl
    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )

    # Note: ChampFlameModel created but unused in pipeline; including for consistency
    model = ChampFlameModel(
            reference_unet,
            reference_control_writer,
            guidance_encoders,
        )
    return guidance_encoders, reference_unet, tokenizer, text_encoder, vae, model


def build_finetune_pipeline(cfg):
    # Load models
    guidance_encoders, reference_unet, tokenizer, text_encoder, vae, model = load_models(cfg)

    # Set up pipeline
    scheduler = DDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler")
    pipeline = StableDiffusionPipeline.from_pretrained(
            cfg.finetuned_model,
            text_encoder=text_encoder,
            vae=vae,
            unet=reference_unet,
            guidance_encoders=guidance_encoders,
            scheduler=scheduler,
            revision=cfg.revision,
        )
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    return pipeline 


def load_baseline_pipeline(
        model_id: str = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
) -> SDPipelineOriginal:
    pipe = SDPipelineOriginal.from_pretrained(model_id, safety_checker=None)
    if torch.cuda.is_available():
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def generate_image(prompt: str,
                   pipe,
                   cfg,
                   steps: int = 300,
                   guidance_scale: float = 9.0) -> Image.Image:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = (torch.Generator(device).manual_seed(cfg.seed)
                 if cfg.seed is not None else None)

    img = pipe(
        prompt=prompt,
        negative_prompt=getattr(cfg, "negative_prompt", NEG_PROMPT_DEFAULT),
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]
    return img

# --------------------------------------------------------------------- #
#                      Miscellaneous utilities                          #
# --------------------------------------------------------------------- #
def clean_prompt(txt: str) -> str:
    return (txt.replace("blurred", "")
                .replace("grainy", "")
                .replace("blurry", "")
                .replace("low-quality", "high-quality")
                .strip())


def load_original_image(meta_key: str, cfg) -> Image.Image:
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG"]:
        path = Path(cfg.original_images) / f"{meta_key}{ext}"
        if path.exists():
            return Image.open(path).convert("RGB")
    return Image.new("RGB", (512, 512), "black")


def count_images_in_directory(directory: Path) -> int:
    """Count valid image files in directory and subdirectories."""
    if not directory.exists():
        return 0

    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    count = 0

    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix in image_extensions:
            count += 1

    return count


def check_directory_populated(directory: Path, expected_count: int, tolerance: float = 0.9) -> bool:
    """Check if directory has at least tolerance * expected_count images."""
    actual_count = count_images_in_directory(directory)
    required_count = int(expected_count * tolerance)
    return actual_count >= required_count


def safe_calculate_fid(path1: str, path2: str, device: str, batch_size: int = 50, min_images: int = 2) -> float:
    """Calculate FID with safety checks for empty directories."""
    # Count images in both directories
    count1 = count_images_in_directory(Path(path1))
    count2 = count_images_in_directory(Path(path2))

    print(f"Directory {path1}: {count1} images")
    print(f"Directory {path2}: {count2} images")

    if count1 < min_images or count2 < min_images:
        print(f"Warning: Not enough images for FID calculation (need at least {min_images} in each directory)")
        return float('inf')

    # Adjust batch size if necessary
    effective_batch_size = min(batch_size, min(count1, count2))
    if effective_batch_size == 0:
        effective_batch_size = 1

    try:
        fid = fid_score.calculate_fid_given_paths(
            [path1, path2],
            batch_size=effective_batch_size,
            device=device,
            dims=2048,
            num_workers=1
        )
        return fid
    except Exception as e:
        print(f"Error calculating FID: {e}")
        return float('inf')


# --------------------------------------------------------------------- #
#                                Main                                   #
# --------------------------------------------------------------------- #
def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_folder = Path(cfg.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    original_dir = output_folder / "original"
    baseline_dir = output_folder / "baseline"
    finetuned_dir = output_folder / "finetuned"

    # Create directories
    for d in [original_dir, baseline_dir, finetuned_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # --------------- Load validation set ----------------------
    val_dict: Dict[str, str] = json.load(open(cfg.validation_dict, "r"))
    expected_image_count = len(val_dict)

    print(f"Expected {expected_image_count} images based on validation dictionary")

    # --------------- Check if directories are already populated ---------------
    original_populated = check_directory_populated(original_dir, expected_image_count)
    baseline_populated = check_directory_populated(baseline_dir, expected_image_count)
    finetuned_populated = check_directory_populated(finetuned_dir, expected_image_count)

    print(f"Original directory populated: {original_populated}")
    print(f"Baseline directory populated: {baseline_populated}")
    print(f"Finetuned directory populated: {finetuned_populated}")

    # --------------- Load generators only if needed -----------------
    baseline_pipe = None
    finetune_pipe = None

    if not baseline_populated or not original_populated:
        print("Loading baseline pipeline...")
        baseline_pipe = load_baseline_pipeline()

    if not finetuned_populated or not original_populated:
        print("Loading finetuned pipeline...")
        finetune_pipe = build_finetune_pipeline(cfg)

    prompts_logged: List[Dict] = []

    # ------------------ Main loop: Generate images -----------------------------
    skip_generation = original_populated and baseline_populated and finetuned_populated

    if skip_generation:
        print("All directories are populated. Skipping image generation...")
        # Still populate prompts_logged for CSV output
        for meta_key, prompt_raw in val_dict.items():
            meta_key = meta_key[:-2]  # strip _0 etc.
            prompt = clean_prompt(prompt_raw)
            prompts_logged.append({"file": meta_key, "prompt": prompt})
    else:
        print("Generating missing images...")
        for meta_key, prompt_raw in val_dict.items():
            meta_key = meta_key[:-2]          # strip _0 etc.
            prompt = clean_prompt(prompt_raw)

            # Check if individual images already exist
            subdir, fname = os.path.split(meta_key)

            orig_path = original_dir / subdir / f"{fname}.png"
            base_path = baseline_dir / subdir / f"{fname}.png"
            fine_path = finetuned_dir / subdir / f"{fname}.png"

            # Create subdirectories
            for d in [original_dir, baseline_dir, finetuned_dir]:
                (d / subdir).mkdir(parents=True, exist_ok=True)

            # Generate only missing images
            if not orig_path.exists():
                img_orig = load_original_image(meta_key, cfg)
                img_orig.save(orig_path)

            if not base_path.exists() and baseline_pipe is not None:
                img_base = generate_image(prompt, baseline_pipe, cfg)
                img_base.save(base_path)

            if not fine_path.exists() and finetune_pipe is not None:
                img_fine = generate_image(prompt, finetune_pipe, cfg)
                img_fine.save(fine_path)

            prompts_logged.append({"file": meta_key, "prompt": prompt})
            print(f"{meta_key:40s} processed.")

    # --------------- Compute FID scores ------------------------------
    print("\nComputing FID scores...")

    fid_base = safe_calculate_fid(
        str(original_dir), 
        str(baseline_dir),
        device=device,
        batch_size=50
    )

    fid_fine = safe_calculate_fid(
        str(original_dir), 
        str(finetuned_dir),
        device=device,
        batch_size=50
    )

    print("\n====================  SUMMARY  ====================")
    if fid_base == float('inf'):
        print(f"FID  (orig ↔ baseline):   Unable to compute (insufficient images)")
    else:
        print(f"FID  (orig ↔ baseline):   {fid_base:8.3f}")

    if fid_fine == float('inf'):
        print(f"FID  (orig ↔ finetune):   Unable to compute (insufficient images)")
    else:
        print(f"FID  (orig ↔ finetune):   {fid_fine:8.3f}")

    print("Lower value indicates images closer to the reference set.")
    print("====================================================")

    # --------------- Save CSV ---------------------------------
    out_csv = output_folder / "fid_prompts.csv"
    pd.DataFrame(prompts_logged).to_csv(out_csv, index=False)
    print(f"Prompt list saved to {out_csv}")


# --------------------------------------------------------------------- #
#                              Entrypoint                               #
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="./configs/inference/flame_emonet_validation.yaml",
                        help="YAML configuration file")
    args = parser.parse_args()

    if not args.config.endswith(".yaml"):
        raise ValueError("Configuration must be a .yaml file")
    cfg = OmegaConf.load(args.config)

    main(cfg)
