import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import DataLoader, Dataset
import inspect

from accelerate import Accelerator
import argparse
from pathlib import Path
from torchvision import transforms
from PIL import Image
import json

prediction_type = "epsilon"
ddpm_num_inference_steps = 1000
ddpm_beta_schedule = 'linear'
ddpm_num_steps = 1000
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
        return {"image": image, "text": description}
    
# Define training function
def train(output_dir, annotation_file, num_epochs,width,height):
    # Load pretrained models
    model_id = "SG161222/RealVisXL_V4.0"
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet",)
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")

    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # Load and prepare dataset
    print(annotation_file)
    dataset = CustomDataset(annotation_file=annotation_file, transform=transform)
    train_dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

    # Initialize accelerator
    accelerator = Accelerator()
    unet, text_encoder, train_dataloader = accelerator.prepare(unet, text_encoder, train_dataloader)
    # Initialize the scheduler
    accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())
    if accepts_prediction_type:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=ddpm_num_steps,
            beta_schedule=ddpm_beta_schedule,
            prediction_type=prediction_type,
        )
    else:
        noise_scheduler = DDPMScheduler(num_train_timesteps=ddpm_num_steps, beta_schedule=ddpm_beta_schedule)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=learning_rate,
        betas=(adam_beta1,adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )
    # Training loop
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            # Get text embeddings
            text_input = tokenizer(batch["text"], padding="max_length", truncation=True, return_tensors="pt").to(accelerator.device)
            text_embeddings = text_encoder(text_input.input_ids)[0]

            # Prepare latents
            latents = torch.randn((batch["image"].shape[0], unet.in_channels, 64, 64))
            latents = latents.to(accelerator.device)
            # figure out the time stepppp
            timesteps = torch.randint(0, 1000, (batch["image"].shape[0],))  # Example: random timestep for each batch element
            timesteps = timesteps.to(accelerator.device)
            # Forward pass
            noise_pred = unet(latents,timesteps, encoder_hidden_states=text_embeddings)

            # Compute loss
            loss = torch.nn.functional.mse_loss(noise_pred, latents)

            # Backward pass and optimization
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Save the fine-tuned model
    pipeline = DiffusionPipeline.from_pretrained(
        model_id,
        unet=accelerator.unwrap_model(unet),
        text_encoder=accelerator.unwrap_model(text_encoder),
    )
    pipeline.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a diffusion model with your dataset")
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the fine-tuned model')
    parser.add_argument('--annotation_file', type=str, default = '/home/vfourel/FaceGPT/Data/LLaVAAnnotations/StableDiffusionPrompts/prompt_response_conversation_All_data.json', help='Path to the annotation JSON file')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs to train the model')
    parser.add_argument('--width', type=int, default=512, help='Width of the training images')
    parser.add_argument('--height', type=int, default=512, help='Height of the training images')
    args = parser.parse_args()

    train(output_dir=args.output_dir, annotation_file = args.annotation_file,  num_epochs=args.num_epochs, width = args.width, height = args.height)
