import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from diffusers import DPMSolverMultistepScheduler
from PIL import Image

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to("cuda")

prompt = "Close up of man going from Angry to Happy"
video_frames = pipe(prompt).frames[0]
video_path = export_to_video(video_frames)
print(video_path)

pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()

# memory optimization
pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
pipe.enable_vae_slicing()

video_frames = pipe(prompt, num_frames=24).frames[0]
video_path = export_to_video(video_frames)
print(video_path)

pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_XL", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# memory optimization
pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
pipe.enable_vae_slicing()

video = [Image.fromarray(frame).resize((1024, 576)) for frame in video_frames]

video_frames = pipe(prompt, video=video, strength=0.6).frames[0]
video_path = export_to_video(video_frames)
print(video_path)