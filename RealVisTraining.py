# Import necessary libraries
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from diffusers import DiffusionPipeline, UNet2DConditionModel, AutoencoderKL
from diffusers.schedulers import DDPMScheduler
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import json
from accelerate import Accelerator

model_id = "SG161222/RealVisXL_V4.0"  # Replace with the actual model path if available
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")

# Define your dataset class
class CustomDataset(Dataset):
    def __init__(self, annotation_file, transform=None):
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        self.transform = transform
        self.image_paths = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        description = self.annotations[image_path]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, description

def train(args):
    # Set up the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model and scheduler
    model = UNet2DConditionModel.from_pretrained("SG161222/RealVisXL_V4.0").to(device)
    vae = AutoencoderKL.from_pretrained("SG161222/RealVisXL_V4.0").to(device)
    scheduler = DDPMScheduler.from_pretrained("SG161222/RealVisXL_V4.0")

    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # Load the dataset
    dataset = CustomDataset(annotation_file=args.annotation_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    mse_loss = torch.nn.MSELoss()

    # Training loop
    model.train()
    for epoch in range(args.epochs):
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            images, descriptions = batch
            optimizer.zero_grad()
            
            images = images.to(device)
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            
            noise = torch.randn_like(latents).to(device)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.size(0),), device=device).long()
            
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            
            predicted_noise = model(noisy_latents, timesteps)["sample"]
            loss = mse_loss(predicted_noise, noise)
            
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {loss.item()}")

    # Save the model
    model.save_pretrained(args.output_dir)
    vae.save_pretrained(args.output_dir)
    print("Model saved to", args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a diffusion model")
    parser.add_argument('--annotation_file', type=str, default = '/home/vfourel/FaceGPT/Data/LLaVAAnnotations/StableDiffusionPrompts/prompt_response_conversation_All_data.json', required=True, help='Path to the annotation JSON file')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save the trained model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--width', type=int, default=512, help='Width of the training images')
    parser.add_argument('--height', type=int, default=512, help='Height of the training images')
    args = parser.parse_args()

    train(args)