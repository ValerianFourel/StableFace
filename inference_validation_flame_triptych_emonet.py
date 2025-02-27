import torch
from models.champ_flame_model import ChampFlameModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL , DDPMScheduler
from models.unet_2d_condition import UNet2DConditionModel
from models.mutual_self_attention import ReferenceAttentionControl
from models.guidance_encoder import GuidanceEncoder
from pipeline.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers import StableDiffusionPipeline as StableDiffusionPipelineOriginal
import argparse
from omegaconf import OmegaConf
import json
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

import torch
from collections import OrderedDict
from pprint import pprint

negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

negative_prompt2 = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"


def create_triptych(original_path, generated_image, finetuned_image, caption, output_path,args):
    # Open the original image

    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG']
    subfolder_and_name = os.path.join(os.path.dirname(original_path), os.path.basename(original_path))
    height = 512

    found_image = False
    for ext in extensions:
        potential_path = os.path.join(args.original_images, subfolder_and_name + ext)
        if os.path.exists(potential_path):
            original_image = Image.open(potential_path)
            original_image = original_image.resize((int(height * original_image.width / original_image.height), height))
            found_image = True
            break

    if not found_image:
        original_image = Image.new('RGB', (512, 512), color='black')

    # Resize all images to have the same height

    original_image = original_image.resize((int(height * original_image.width / original_image.height), height))
    generated_image = generated_image.resize((512, 512))
    finetuned_image = finetuned_image.resize((512, 512))
    
    # Create a new image with the combined width and some padding
    total_width = original_image.width + generated_image.width + finetuned_image.width + 40  # 20px padding between images
    triptych = Image.new('RGB', (total_width, height + 60))  # 60px extra height for caption
    
    # Paste the images
    triptych.paste(original_image, (0, 0))
    triptych.paste(generated_image, (original_image.width + 20, 0))
    triptych.paste(finetuned_image, (original_image.width + generated_image.width + 40, 0))
    
    # Add caption
    draw = ImageDraw.Draw(triptych)
    font = ImageFont.load_default()
    draw.text((10, height + 10), caption, fill=(255, 255, 255), font=font)
    
    # Save the triptych
    triptych.save(output_path)


# Usage


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


def load_model_original_pipeline(model_id="SG161222/Realistic_Vision_V6.0_B1_noVAE", device="cuda"):
    # Load the model pipeline
    pipeline = StableDiffusionPipelineOriginal.from_pretrained(model_id)
        # Remove the safety checker
    pipeline.safety_checker = None
    # Move to the appropriate device
    if torch.cuda.is_available() and device == "cuda":
        pipeline = pipeline.to("cuda")
    else:
        pipeline = pipeline.to("cpu")
    
    return pipeline

def generate_image(prompt, pipeline, args, num_inference_steps=300, guidance_scale=9.0):
    # Generate an image from the prompt with an optional negative prompt
    # Set up generator for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None
    with torch.no_grad():
        #prompt = "high detail, 4k, photorealistic" + prompt
        image = pipeline(prompt=prompt, 
                      negative_prompt=args.negative_prompt, 
                      num_inference_steps=num_inference_steps, 
                      guidance_scale=guidance_scale,
                       generator=generator).images[0]
    return image

def save_image(image, path="random.png"):
    # Save the image to the specified path
    image.save(path)

def clean_string(value):
    return value.replace("blurred", "").replace("grainy", "").replace("blurry", "").replace("low-quality", "high-quality").strip()

def main(args):
 
    # Load the pipeline
    pipeline_original = load_model_original_pipeline()

    pipeline = inference_pipeline(args)

        # Path to your JSON file
    json_file_path = args.validation_dict

    # Read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Create validation folder if it doesn't exist
    folder = args.output_folder
    os.makedirs(folder, exist_ok=True)

    for i, (key, value) in enumerate(data.items(), start=1):
        prompt = value[0]
                # Split the key into subfolder and filename
        subfolder, filename = os.path.split(key)

        # Create the complete output path
        output_path = os.path.join(folder, subfolder)
        os.makedirs(output_path, exist_ok=True)
        # Generate the images
        prompt  = clean_string(prompt)
        image_original = generate_image(prompt, pipeline_original,args)
        image_finetune = generate_image(prompt, pipeline,args)

        # Create output filename with .png extension
        output_filename = os.path.join(output_path, f"{filename}.png")
        create_triptych(key, image_original, image_finetune, prompt, output_filename,args)

        print(f"Triptych generated and saved as '{output_filename}'")

    print("All triptychs have been generated and saved in the 'validation' folder.")

if __name__ == "__main__":
    import shutil
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/inference/flame_emonet_validation.yaml")
    args = parser.parse_args()

    if args.config.endswith(".yaml"):
        args = OmegaConf.load(args.config)
    else:
        raise ValueError("Do not support this format config file")
    main(args)