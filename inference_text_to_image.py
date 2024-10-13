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

import torch
from collections import OrderedDict
from pprint import pprint

negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

negative_prompt2 = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"

def inspect_model_file(file_path):
    # Load the state dict
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the state dict
    state_dict = torch.load(file_path, map_location=device)
    # Check if it's an OrderedDict (typical for model state dicts)
    if isinstance(state_dict, OrderedDict):
        print("File contains a state dict (OrderedDict)")
        
        # Get basic info
        num_keys = len(state_dict)
        print(f"Number of keys: {num_keys}")
        
        # Print the first few keys and their tensor shapes
        print("\nFirst 10 keys and their tensor shapes:")
        for i, (key, tensor) in enumerate(state_dict.items()):
            if i >= 10:
                break
            print(f"{key}: {tensor.shape}")
        
        # Get all unique top-level keys (assuming the format "layer.sublayer.weight")
        top_level_keys = set(key.split('.')[0] for key in state_dict.keys())
        print("\nTop-level keys (potential model components):")
        pprint(list(top_level_keys))
        
    else:
        print("File does not contain a standard PyTorch state dict")
        print("Content type:", type(state_dict))
        
        # If it's a dict-like object, try to print its keys
        if hasattr(state_dict, 'keys'):
            print("\nKeys in the object:")
            pprint(list(state_dict.keys()))
        else:
            print("Unable to inspect the contents further.")

# Usage


def load_guidance_encoder(cfg):
    # Define the path to the pre-trained weights
    pretrained_path = "/ps/scratch/ps_shared/vfourel/ChampFace/final-sd-model-finetuned-l192_lpips08-snr08-lr56-1024pics_224res/checkpoint-176/flame_encoder/guidance_encoder_flame.pth"
    
    # Initialize the GuidanceEncoder
    guidance_encoder_flame = GuidanceEncoder(guidance_embedding_channels=cfg.guidance_encoder_kwargs.guidance_embedding_channels,
            guidance_input_channels=cfg.guidance_encoder_kwargs.guidance_input_channels,
            block_out_channels=cfg.guidance_encoder_kwargs.block_out_channels,)
    
    # Load the pre-trained weights
    state_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    
    # Check if the loaded state_dict is wrapped (e.g., with DataParallel)
    if "module." in list(state_dict.keys())[0]:
        # Remove the "module." prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # Load the weights into the model
    guidance_encoder_flame.load_state_dict(state_dict)
    
    # Move the model to the appropriate device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    guidance_encoder_flame = guidance_encoder_flame.to(device)
    
    # Set the model to evaluation mode
    guidance_encoder_flame.eval()
    guidance_encoder_group = dict()

    for guidance_type in cfg.data.guids:
        guidance_encoder_group[guidance_type] = guidance_encoder_flame
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
    guidance_encoder_flame = load_guidance_encoder(args)
    
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
            guidance_encoder_flame,
        )
    return guidance_encoder_flame, reference_unet, tokenizer, text_encoder, vae, model

def run_inference_ChampModel(args,negative_prompt=negative_prompt):
    guidance_encoder_flame, reference_unet, tokenizer, text_encoder, vae, model = load_models(args)
    demonstrationPaths = "/home/vfourel/FaceGPT/Data/FlameImagesAffectnet/demonstration.json"



def run_inference_pipeline(args,negative_prompt=negative_prompt):
    # Load models
    guidance_encoder_flame, reference_unet, tokenizer, text_encoder, vae, model = load_models(args)
    
    # Set up pipeline
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    pipeline = StableDiffusionPipeline.from_pretrained(
            args.finetuned_model,
            text_encoder=text_encoder, # we have to 
            vae=vae,
            unet=reference_unet,
            guidance_encoder_flame=guidance_encoder_flame,
            revision=args.revision,
        )
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = pipeline.to(device)
    
    # Set up generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None
    
    # Run inference
    prompt = "Disgust, Man"
    negative_prompt = negative_prompt
    multi_guidance_lst = "Your multi-guidance list here"  # Adjust based on your model's requirements
    
    with torch.no_grad():
        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            multi_guidance_lst=multi_guidance_lst,
            num_inference_steps=20,
            generator=generator
        ).images[0]
    
    # Save or display the image
    return image


def load_model_original(model_id="SG161222/Realistic_Vision_V6.0_B1_noVAE", device="cuda"):
    # Load the model pipeline
    pipeline = StableDiffusionPipelineOriginal.from_pretrained(model_id)
    
    # Move to the appropriate device
    if torch.cuda.is_available() and device == "cuda":
        pipeline = pipeline.to("cuda")
    else:
        pipeline = pipeline.to("cpu")
    
    return pipeline

def generate_image(prompt, model, num_inference_steps=50, guidance_scale=7.5, negative_prompt=negative_prompt):
    # Generate an image from the prompt with an optional negative prompt
    with torch.no_grad():
        image = model(prompt=prompt, 
                      negative_prompt=negative_prompt, 
                      num_inference_steps=num_inference_steps, 
                      guidance_scale=guidance_scale).images[0]
    return image

def save_image(image, path="output2.png"):
    # Save the image to the specified path
    image.save(path)

def main(args):
    #file_path = "/ps/scratch/ps_shared/vfourel/ChampFace/sd-model-finetuned-l1-snr50-lr05/checkpoint-128/unet/diffusion_pytorch_model.safetensors"
    #inspect_model_file(file_path)
    # Define your prompt
    prompt = "A photorealistic painting of a futuristic city at sunset, with flying cars and neon lights"
    prompt = "Man, Disgust"
    # Load the model
    model = load_model_original()

    # Generate the image
    image = generate_image(prompt, model)

    # Save the image
    save_image(image, "generated_image6.png")

    print("Image generated and saved as 'generated_image5.png'")

    image_finetune = run_inference_pipeline(args)

    save_image(image_finetune, "image_finetune7.png")

    print("Image generated and saved as 'image_finetune2.png'")

if __name__ == "__main__":
    import shutil
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/inference/flame_inference.yaml")
    args = parser.parse_args()

    if args.config.endswith(".yaml"):
        args = OmegaConf.load(args.config)
    else:
        raise ValueError("Do not support this format config file")
    main(args)
