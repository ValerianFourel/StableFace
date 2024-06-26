# Import necessary libraries
import argparse
from pathlib import Path
import torch
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from PIL import Image

# Constants
MODEL_noVAE = "SG161222/RealVisXL_V4.0"  # Replace with the actual model path if available

class Predictor:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_count = 1  # Initialize image counter
        self.log_file = self.output_dir / "generation.log"  # Define log file path

    def setup(self):
        self.rv_VAE = DiffusionPipeline.from_pretrained(
            MODEL_noVAE,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str,
        negative_prompt: str,
        scheduler: str = "DDIM",
        width: int = 512,
        height: int = 768,
        guidance_scale: int = 7,
        num_inference_steps: int = 20,
        seed: int = 42,
        number_picture: int = 1,
    ):
        generator = torch.Generator("cuda").manual_seed(seed)
        parameters = {
            "prompt": [prompt] * number_picture,
            "negative_prompt": [negative_prompt] * number_picture,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator
        }

        self.rv_VAE.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.rv_VAE.scheduler.config
        )

        images = self.rv_VAE(**parameters)
        output = []
        for i, sample in enumerate(images.images):
            filename = f"picture_{self.image_count:05d}.png"
            output_path = self.output_dir / filename
            sample.save(output_path)
            output.append(output_path)
            with self.log_file.open("a") as log:
                log.write(f"{filename}: prompt: '{prompt}', negative_prompt: '{negative_prompt}'\n")
            self.image_count += 1

        return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images with a diffusion model")
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save the generated images')
    args = parser.parse_args()

    predictor = Predictor(output_dir=args.output_dir)
    predictor.setup()

    print("Enter 'quit' to exit the program at any time.")
    while True:
        prompt = input("Enter your prompt: ")
        if prompt.lower() == 'quit':
            break
        negative_prompt = input("Enter your negative prompt: ")
        if negative_prompt.lower() == 'quit':
            break

        result_paths = predictor.predict(prompt=prompt, negative_prompt=negative_prompt)
        for path in result_paths:
            print(f"Generated image saved to {path}")
