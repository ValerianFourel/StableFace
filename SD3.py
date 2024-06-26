import torch
# from diffusers import StableDiffusionPipeline
import diffusers 
pipe = diffusers.StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

image = pipe(
    "A cat holding a sign that says hello world",
    negative_prompt="",
    num_inference_steps=28,
    guidance_scale=7.0,
).images[0]

# Save the image
image.save("generated_image.png")
