#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate images (baseline + finetuned) and evaluate the finetuned result with
EmoNet (valence, arousal, emotion class).  
Top-1 / Top-3 accuracy is computed w.r.t. the ground-truth label defined as the
first word in the prompt.

Dependencies
------------
pip install diffusers transformers omegaconf pandas pillow torchvision \
            scikit-image opencv-python emonet-pytorch
"""

# --------------------------------------------------------------------- #
#                                Imports                                #
# --------------------------------------------------------------------- #
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from skimage import io
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# Stable-Diffusion stack -------------------------------------------------
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers import StableDiffusionPipeline as SDPipelineOriginal
from transformers import CLIPTokenizer, CLIPTextModel

# project-specific modules ----------------------------------------------
from models.champ_flame_model import ChampFlameModel
from models.guidance_encoder import GuidanceEncoder
from models.mutual_self_attention import ReferenceAttentionControl
from models.unet_2d_condition import UNet2DConditionModel
from pipeline.pipeline_stable_diffusion import StableDiffusionPipeline

# EmoNet -----------------------------------------------------------------
from external.emonet.emonet.models import EmoNet
# ----------------------------------------------------------------------- #


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

EMO_LABELS_8 = [
    "neutral", "happy", "sad", "surprise",
    "fear", "disgust", "anger", "contempt"
]
EMO_LABELS_5 = [
    "neutral", "happy", "sad", "surprise", "anger"
]
# --------------------------------------------------------------------- #


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

def load_models(args):
        # Load tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    
    # Load UNet
    reference_unet = UNet2DConditionModel.from_pretrained(args.finetuned_model, subfolder="unet")
    
    # Setup guidance encoder
    guidance_encoders = load_guidance_encoder(args)
    
    # Create ReferenceAttentionControl
    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    
    model = ChampFlameModel(
            reference_unet,
            reference_control_writer,
            guidance_encoders,
        )
    return guidance_encoders, reference_unet, tokenizer, text_encoder, vae, model


def inference_pipeline(args):
    # Load models
    guidance_encoders, reference_unet, tokenizer, text_encoder, vae, model = load_models(args)
    
    # Set up pipeline
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    pipeline = StableDiffusionPipeline.from_pretrained(
            args.finetuned_model,
            text_encoder=text_encoder, # we have to 
            vae=vae,
            unet=reference_unet,
            guidance_encoders=guidance_encoders,
            revision=args.revision,
        )
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = pipeline.to(device)
    
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


# --------------------------------------------------------------------- #
#                            EmoNet helpers                             #
# --------------------------------------------------------------------- #
def load_emonet(n_classes: int,
                device: str = "cuda") -> Tuple[EmoNet, transforms.Compose, List[str]]:
    """
    Load EmoNet checkpoint (5 or 8 classes).
    """
    ckpt_name = f"emonet_{n_classes}.pth"
    ckpt_path = f"./external/emonet/pretrained/{ckpt_name}"
    #if not ckpt_path.exists():
        # fall back to package helper (downloads if necessary)
     #   ckpt_path = EmoNet.get_checkpoint(f"emonet_{n_classes}")

    state_dict = torch.load(ckpt_path, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model = EmoNet(n_expression=n_classes).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    labels = EMO_LABELS_8 if n_classes == 8 else EMO_LABELS_5

    tfms = transforms.Compose([
    transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor()            # converts H×W×C uint8 [0-255] → C×H×W float32 [0-1]
    ])
    return model, tfms, labels


@torch.no_grad()
def emonet_infer(img: Image.Image,
                 model: EmoNet,
                 tfms: transforms.Compose,
                 labels: List[str],
                 device: str = "cuda") -> Tuple[str, List[str], float, float]:
    """
    Returns:
        top1_label
        top3_labels
        valence   (clamped to [-1,1])
        arousal   (clamped to [-1,1])
    """
    tensor = tfms(img.convert("RGB")).unsqueeze(0).to(device)
    out = model(tensor)

    expr_logits = out["expression"]
    valence = float(out["valence"].clamp(-1.0, 1.0))
    arousal = float(out["arousal"].clamp(-1.0, 1.0))

    probs = torch.softmax(expr_logits, 1).squeeze()
    top3 = torch.topk(probs, 3).indices.cpu().tolist()
    top3_labels = [labels[i] for i in top3]
    return top3_labels[0], top3_labels, valence, arousal
# --------------------------------------------------------------------- #


# --------------------------------------------------------------------- #
#                          Misc. utilities                              #
# --------------------------------------------------------------------- #
def clean_prompt(txt: str) -> str:
    return (txt.replace("blurred", "")
                .replace("grainy", "")
                .replace("blurry", "")
                .replace("low-quality", "high-quality")
                .strip())


def create_triptych(orig_meta_path: str,
                    baseline_img: Image.Image,
                    finetune_img: Image.Image,
                    caption: str,
                    out_png: Path,
                    cfg):
    """
    (original | baseline | finetuned) concatenation.
    """
    h = 512
    extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG"]
    original_found = False

    for ext in extensions:
        candidate = Path(cfg.original_images) / f"{orig_meta_path}{ext}"
        if candidate.exists():
            original_img = Image.open(candidate).convert("RGB")
            w = int(h * original_img.width / original_img.height)
            original_img = original_img.resize((w, h))
            original_found = True
            break

    if not original_found:
        original_img = Image.new("RGB", (512, 512), "black")

    baseline_img = baseline_img.resize((512, 512))
    finetune_img = finetune_img.resize((512, 512))

    total_w = original_img.width + baseline_img.width + finetune_img.width + 40
    canvas = Image.new("RGB", (total_w, h + 60), "black")

    canvas.paste(original_img, (0, 0))
    canvas.paste(baseline_img, (original_img.width + 20, 0))
    canvas.paste(finetune_img, (original_img.width + baseline_img.width + 40, 0))

    draw = ImageDraw.Draw(canvas)
    draw.text((10, h + 10), caption, font=ImageFont.load_default(),
              fill=(255, 255, 255))
    canvas.save(out_png)
# --------------------------------------------------------------------- #

# --------------------------------------------------------------------- #
#                         helper: original image                        #
# --------------------------------------------------------------------- #
def load_original_image(meta_key: str, cfg) -> Image.Image:
    """Return the PIL image corresponding to meta_key or a black placeholder."""
    extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG"]
    for ext in extensions:
        path = Path(cfg.original_images) / f"{meta_key}{ext}"
        if path.exists():
            return Image.open(path).convert("RGB")
    # fallback (no file found)
    return Image.new("RGB", (512, 512), "black")


# --------------------------------------------------------------------- #
#                                Main                                   #
# --------------------------------------------------------------------- #
def main(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(cfg.output_folder).mkdir(parents=True, exist_ok=True)

    # ------------------ Load all models ------------------
    baseline_pipe = load_baseline_pipeline()
    finetune_pipe = inference_pipeline(cfg)
    emo_model, emo_tfms, emo_labels = load_emonet(
        getattr(cfg, "nclasses", 8), device)

    # ----------------- Load validation set ---------------
    with open(cfg.validation_dict, "r") as f:
        val_dict = json.load(f)

    # ----------------------- Stats -----------------------
    stats = {
        "base_top1": 0, "base_top2": 0, "base_top3": 0,
        "fine_top1": 0, "fine_top2": 0, "fine_top3": 0,
        "same_top1": 0, "same_top3": 0,
        "total": 0
    }
    csv_rows: List[Dict] = []

    # ------------------ Processing loop ------------------
    # random_items = dict(random.sample(list(val_dict.items()), 500))

    for meta_key, value in val_dict.items():
        meta_key = meta_key[:-2]

        prompt_raw = value
        print(value)
        gt_label = prompt_raw.split()[0].lower()
        prompt = clean_prompt(prompt_raw)

        # ---- load original image ----
        img_orig = load_original_image(meta_key, cfg)

        # ---- generate baseline / finetuned ----
        img_base = generate_image(prompt, baseline_pipe, cfg)
        img_fine = generate_image(prompt, finetune_pipe, cfg)

        # ---- EmoNet inference (orig / base / fine) ----
        p1_o, p3_o, val_o, aro_o = emonet_infer(
            img_orig, emo_model, emo_tfms, emo_labels, device)
        p1_b, p3_b, val_b, aro_b = emonet_infer(
            img_base, emo_model, emo_tfms, emo_labels, device)
        p1_f, p3_f, val_f, aro_f = emonet_infer(
            img_fine, emo_model, emo_tfms, emo_labels, device)

        # ---- update stats (baseline & finetuned only) ---
        stats["total"] += 1

        # baseline rank accuracy
        if gt_label == p1_b:
            stats["base_top1"] += 1
        elif gt_label == p3_b[1]:
            stats["base_top2"] += 1
        elif gt_label in p3_b:
            stats["base_top3"] += 1

        # finetuned rank accuracy
        if gt_label == p1_f:
            stats["fine_top1"] += 1
        elif gt_label == p3_f[1]:
            stats["fine_top2"] += 1
        elif gt_label in p3_f:
            stats["fine_top3"] += 1

        # overlap between pipelines
        stats["same_top1"] += int(p1_b == p1_f)
        stats["same_top3"] += int(bool(set(p3_b) & set(p3_f)))

        # ---- triptych (unchanged) -----------------------
        subdir, fname = os.path.split(meta_key)
        (Path(cfg.output_folder) / subdir).mkdir(parents=True, exist_ok=True)
        trip_png = Path(cfg.output_folder) / subdir / f"{fname}.png"
        create_triptych(meta_key, img_base, img_fine, prompt, trip_png, cfg)

        # ---- CSV row -----------------------------------
        csv_rows.append({
            "file":        meta_key,
            "prompt":      prompt,
            "gt":          gt_label,

            # original image
            "orig_top1":   p1_o,
            "orig_top2":   p3_o[1],
            "orig_top3":   p3_o[2],
            "orig_val":    val_o,
            "orig_aro":    aro_o,

            # baseline
            "base_top1":   p1_b,
            "base_top2":   p3_b[1],
            "base_top3":   p3_b[2],
            "base_val":    val_b,
            "base_aro":    aro_b,
            "base_rank":   p3_b.index(gt_label) + 1 if gt_label in p3_b else 0,

            # finetuned
            "fine_top1":   p1_f,
            "fine_top2":   p3_f[1],
            "fine_top3":   p3_f[2],
            "fine_val":    val_f,
            "fine_aro":    aro_f,
            "fine_rank":   p3_f.index(gt_label) + 1 if gt_label in p3_f else 0,

            # consistency
            "same_top1":   int(p1_b == p1_f),
            "same_top3":   int(bool(set(p3_b) & set(p3_f)))
        })

        print(f"{meta_key:50s} gt={gt_label:<9}"
              f" | O:{p1_o:<9}"
              f" | B:{p1_b:<9}/{p3_b[1]:<9}/{p3_b[2]:<9}"
              f" | F:{p1_f:<9}/{p3_f[1]:<9}/{p3_f[2]:<9}")

    # ----------------------- Summary ---------------------
    t = stats["total"]
    pct = lambda x: x / t * 100

    print("\n================  ACCURACY per RANK  ================")
    print(f"Baseline   : Top-1 {pct(stats['base_top1']):6.2f}% "
          f"Top-2 {pct(stats['base_top2']):6.2f}% "
          f"Top-3 {pct(stats['base_top3']):6.2f}%")
    print(f"Finetuned  : Top-1 {pct(stats['fine_top1']):6.2f}% "
          f"Top-2 {pct(stats['fine_top2']):6.2f}% "
          f"Top-3 {pct(stats['fine_top3']):6.2f}%")
    print("------------------------------------------------------")
    print(f"Same Top-1 prediction  : {pct(stats['same_top1']):6.2f}%")
    print(f"Any overlap in Top-3   : {pct(stats['same_top3']):6.2f}%")
    print("======================================================")

    # ------------------- CSV output ----------------------
    df = pd.DataFrame(csv_rows)
    csv_path = Path(cfg.output_folder) / "emotion_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to {csv_path}")


# --------------------------------------------------------------------- #
#                              Entrypoint                               #
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="./configs/inference/flame_emonet_validation.yaml",
                        help="YAML configuration file")
    cli = parser.parse_args()

    if cli.config.endswith(".yaml"):
        cfg = OmegaConf.load(cli.config)
    else:
        raise ValueError("Configuration must be a .yaml file")

    main(cfg)
