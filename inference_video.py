import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from PIL import Image

# Replace this with the actual path to your file
file_path = "/ps/scratch/ps_shared/vfourel/affectnet_41k_AffectOnly/EmocaProcessed_38k/inputs/810/a1eee929d8804c0b6f3d6f3bb5bed336fa0cbfdf1e8103e974dee1b200.png"

# Open the image
image = Image.open(file_path)

# Initialize the pipeline
pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.enable_model_cpu_offload()

# Set the prompt
prompt = image

# Generate the video
video_frames = pipeline(
    prompt,
    num_inference_steps=50,
    num_frames=16,
    height=576,
    width=1024,
).frames

# Export the video
export_to_video(video_frames, "man_angry_to_happy.mp4")

print("Video generation complete. Output saved as 'man_angry_to_happy.mp4'")
