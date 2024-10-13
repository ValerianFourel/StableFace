import torch
import torchvision.utils as vutils
import os
from pathlib import Path
import math

def tensor_to_grid_picture(tensor, output_folder, filename="grid_picture.png"):
    """
    Transform a tensor of shape (N, 3, H, W) into a large collage of all pictures
    and save it as a PNG file in the specified folder.
    
    Args:
    tensor (torch.Tensor): Input tensor of shape (N, 3, H, W)
    output_folder (str): Path to the folder where the image will be saved
    filename (str, optional): Name of the output file. Defaults to "grid_picture.png"
    
    Returns:
    str: Path to the saved image file
    """
    # Check input shape
    if len(tensor.shape) != 4 or tensor.shape[1] != 3:
        raise ValueError(f"Expected input shape (N, 3, H, W), but got {tensor.shape}")
    
    # Normalize the tensor to [0, 1] range if it's not already
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    
    # Create a grid from all the images
    num_images = tensor.shape[0]
    nrow = int(math.ceil(math.sqrt(num_images)))  # Calculate number of rows for a square-like grid
    grid = vutils.make_grid(tensor, nrow=nrow, padding=2, normalize=False)
    
    # Ensure the output folder exists
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Full path for the output file
    output_file = output_path / filename
    
    # Save the grid picture as PNG
    vutils.save_image(grid, output_file)
    
    return str(output_file)

