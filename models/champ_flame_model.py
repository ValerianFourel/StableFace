#####################################
# Made by VF
# modifed from : https://github.com/fudan-generative-vision/champ/blob/master/models/champ_model.py
#


import torch
import torch.nn as nn
from models.unet_2d_condition import UNet2DConditionModel
import os
import json
import shutil



class ChampFlameModel(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        # denoising_unet: UNet3DConditionModel,
        reference_control_writer,
        # reference_control_reader,
        guidance_encoder_group,
    ):
        
        super().__init__()
        self.reference_unet = reference_unet
        # self.denoising_unet = denoising_unet

        self.reference_control_writer = reference_control_writer
       # self.reference_control_reader = reference_control_reader

        self.guidance_types = []
        self.guidance_input_channels = []

#########################
# VF : you need to look for the guidance input
        for guidance_type, guidance_module in guidance_encoder_group.items():
            setattr(self, f"guidance_encoder_{guidance_type}", guidance_module)
            self.guidance_types.append(guidance_type)
            self.guidance_input_channels.append(guidance_module.guidance_input_channels)
        
        self.guidance_input_channels_flame = self.guidance_input_channels[0]
        self.flame_encoder = guidance_encoder_group.get(self.guidance_types[0], None)
    
        
    
      

    def save_model(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the main model's state dict
        # torch.save(self.state_dict(), os.path.join(save_directory, "champ_flame_model.pt"))
        
        # Save the flame_encoder in a separate subfolder
        flame_encoder_dir = os.path.join(save_directory, "flame_encoder")
        os.makedirs(flame_encoder_dir, exist_ok=True)
        for guidance_type in self.guidance_types:
            flame_encoder = getattr(self, f"guidance_encoder_{guidance_type}")
            torch.save(flame_encoder.state_dict(), os.path.join(flame_encoder_dir, f"guidance_encoder_{guidance_type}.pth"))
        
        # Save the reference UNet model directly as diffusion_pytorch_model.bin
        unet_save_path = save_directory # os.path.join(save_directory, "unet")
        os.makedirs(unet_save_path, exist_ok=True)
        torch.save(self.reference_unet.state_dict(), os.path.join(unet_save_path, "diffusion_pytorch_model.bin"))

        # Save the model's config
        config = {
            "model_type": "ChampFlameModel",
            "guidance_types": self.guidance_types,
            "guidance_input_channels": self.guidance_input_channels,
            # Add any other relevant configuration details
        }
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        print(f"ChampFlameModel and its components saved to {save_directory}")
    
    def save_pretrained(self, save_directory):
        # Create the directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)
                # Save the reference UNet model directly as diffusion_pytorch_model.bin
        unet_save_path = save_directory # os.path.join(save_directory, "unet")
        os.makedirs(unet_save_path, exist_ok=True)
        torch.save(self.reference_unet.state_dict(), os.path.join(unet_save_path, "diffusion_pytorch_model.bin"))

        # Save the reference UNet model
        #self.reference_unet.save_pretrained(save_directory)


        # Save the flame encoder in a 'flame_encoder' subfolder in the parent directory of save_directory
        parent_directory = os.path.dirname(save_directory)
        flame_encoder_save_path = os.path.join(parent_directory, 'flame_encoder')
        os.makedirs(flame_encoder_save_path, exist_ok=True)
        
        # Save the flame encoder model
        if self.flame_encoder is not None:
            torch.save(self.flame_encoder.state_dict(), os.path.join(flame_encoder_save_path, 'pytorch_model.bin'))
        else:
            raise AttributeError("Flame encoder module not found.")
        
        print(f"Model saved to {save_directory}")

    def forward(
        self,
        noisy_latents,
        timesteps,
        encoder_hidden_states,
        # ref_image_latents,
        multi_guidance_cond,
      #   uncond_fwd: bool = False,
    ):
        guidance_cond_group = torch.split(
            multi_guidance_cond, self.guidance_input_channels, dim=1
        )
        guidance_fea_lst = []

        for guidance_idx, guidance_cond in enumerate(guidance_cond_group):
            guidance_encoder = getattr(
                self, f"guidance_encoder_{self.guidance_types[guidance_idx]}"
            )
            guidance_fea = guidance_encoder(guidance_cond)
            guidance_fea_lst += [guidance_fea]

        guidance_fea = torch.stack(guidance_fea_lst, dim=0).sum(0)

        model_pred = self.reference_unet( # VF: we just want to use the reference unet 
            noisy_latents,
            timesteps,
            encoder_hidden_states= encoder_hidden_states,
            guidance_fea=guidance_fea,
            # added_cond_kwargs= {
            #     "text_embeds": encoder_hidden_states , # You need to provide this
            #     "time_ids":timesteps
            # }, 
       #     encoder_hidden_states=clip_image_embeds,
        )

        return model_pred