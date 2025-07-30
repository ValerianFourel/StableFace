#!/usr/bin/env python3
"""
Standalone FID Calculator for Nested Directory Structures

This script computes FID scores between directories that contain images in subfolders.
It handles the nested structure properly and includes robust error handling.

Usage:
    python compute_fid.py --original /path/to/original --baseline /path/to/baseline --finetuned /path/to/finetuned
"""

import argparse
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple
import torch
from PIL import Image
from pytorch_fid import fid_score


def find_all_images(directory: Path) -> List[Path]:
    """Find all image files in directory and subdirectories."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = []

    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix in image_extensions:
            image_files.append(file_path)

    return image_files


def validate_image(image_path: Path) -> bool:
    """Check if image file is valid and can be opened."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        print(f"Invalid image {image_path}: {e}")
        return False


def create_flat_temp_directory(source_dir: Path, temp_base: str) -> Tuple[str, int]:
    """
    Create a temporary directory with all valid images flattened (no subfolders).
    This helps pytorch-fid handle the images properly.
    """
    temp_dir = tempfile.mkdtemp(prefix=temp_base)
    temp_path = Path(temp_dir)

    image_files = find_all_images(source_dir)
    valid_count = 0

    print(f"Processing {len(image_files)} images from {source_dir}")

    for i, img_file in enumerate(image_files):
        if validate_image(img_file):
            # Create a unique filename to avoid conflicts
            dest_name = f"img_{i:06d}{img_file.suffix}"
            dest_path = temp_path / dest_name

            try:
                shutil.copy2(img_file, dest_path)
                valid_count += 1
            except Exception as e:
                print(f"Failed to copy {img_file}: {e}")

        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(image_files)} images")

    print(f"Created temp directory with {valid_count} valid images: {temp_dir}")
    return temp_dir, valid_count


def compute_fid_robust(path1: str, path2: str, device: str = "cuda") -> float:
    """
    Compute FID with robust error handling and progressive batch size reduction.
    """
    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"
        print("CUDA not available, using CPU")

    # Try different batch sizes
    batch_sizes = [50, 32, 16, 8, 4, 2, 1]

    for batch_size in batch_sizes:
        try:
            print(f"Attempting FID calculation with batch_size={batch_size}")
            fid = fid_score.calculate_fid_given_paths(
                [path1, path2],
                batch_size=batch_size,
                device=device,
                dims=2048,
                num_workers=0  # Disable multiprocessing to avoid issues
            )
            print(f"✓ FID calculation successful with batch_size={batch_size}")
            return fid
        except Exception as e:
            print(f"✗ Failed with batch_size={batch_size}: {e}")
            continue

    print("All batch sizes failed!")
    return float('inf')


def main():
    parser = argparse.ArgumentParser(description="Compute FID scores for nested directory structures")
    parser.add_argument("--original", required=True, help="Path to original images directory")
    parser.add_argument("--baseline", required=True, help="Path to baseline generated images directory") 
    parser.add_argument("--finetuned", required=True, help="Path to finetuned generated images directory")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to use for computation")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary directories for debugging")

    args = parser.parse_args()

    # Validate input directories
    original_dir = Path(args.original)
    baseline_dir = Path(args.baseline)
    finetuned_dir = Path(args.finetuned)

    for name, path in [("Original", original_dir), ("Baseline", baseline_dir), ("Finetuned", finetuned_dir)]:
        if not path.exists():
            print(f"Error: {name} directory does not exist: {path}")
            return
        if not path.is_dir():
            print(f"Error: {name} path is not a directory: {path}")
            return

    print("=" * 60)
    print("FID COMPUTATION FOR NESTED DIRECTORIES")
    print("=" * 60)

    # Count images in each directory
    original_images = find_all_images(original_dir)
    baseline_images = find_all_images(baseline_dir)
    finetuned_images = find_all_images(finetuned_dir)

    print(f"Found {len(original_images)} images in original directory")
    print(f"Found {len(baseline_images)} images in baseline directory") 
    print(f"Found {len(finetuned_images)} images in finetuned directory")

    if len(original_images) < 2 or len(baseline_images) < 2 or len(finetuned_images) < 2:
        print("Error: Need at least 2 images in each directory for FID calculation")
        return

    # Create temporary flattened directories
    print("\nCreating temporary flattened directories...")
    temp_dirs = []

    try:
        # Create flattened temporary directories
        temp_original, valid_original = create_flat_temp_directory(original_dir, "fid_original_")
        temp_baseline, valid_baseline = create_flat_temp_directory(baseline_dir, "fid_baseline_")
        temp_finetuned, valid_finetuned = create_flat_temp_directory(finetuned_dir, "fid_finetuned_")

        temp_dirs = [temp_original, temp_baseline, temp_finetuned]

        print(f"\nValid images: Original={valid_original}, Baseline={valid_baseline}, Finetuned={valid_finetuned}")

        if valid_original < 2 or valid_baseline < 2 or valid_finetuned < 2:
            print("Error: Not enough valid images after validation")
            return

        # Compute FID scores
        print("\n" + "=" * 60)
        print("COMPUTING FID SCORES")
        print("=" * 60)

        print("\n1. Computing FID: Original ↔ Baseline")
        fid_baseline = compute_fid_robust(temp_original, temp_baseline, args.device)

        print("\n2. Computing FID: Original ↔ Finetuned")
        fid_finetuned = compute_fid_robust(temp_original, temp_finetuned, args.device)

        # Display results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)

        if fid_baseline != float('inf'):
            print(f"FID (Original ↔ Baseline):   {fid_baseline:8.3f}")
        else:
            print(f"FID (Original ↔ Baseline):   FAILED TO COMPUTE")

        if fid_finetuned != float('inf'):
            print(f"FID (Original ↔ Finetuned):  {fid_finetuned:8.3f}")
        else:
            print(f"FID (Original ↔ Finetuned):  FAILED TO COMPUTE")

        print("\nNote: Lower FID values indicate better quality/similarity to original images")

        if fid_baseline != float('inf') and fid_finetuned != float('inf'):
            if fid_finetuned < fid_baseline:
                improvement = fid_baseline - fid_finetuned
                print(f"✓ Finetuned model is better by {improvement:.3f} FID points")
            elif fid_baseline < fid_finetuned:
                degradation = fid_finetuned - fid_baseline
                print(f"✗ Baseline model is better by {degradation:.3f} FID points")
            else:
                print("→ Both models perform equally")

    finally:
        # Clean up temporary directories
        if not args.keep_temp:
            print(f"\nCleaning up temporary directories...")
            for temp_dir in temp_dirs:
                if temp_dir and Path(temp_dir).exists():
                    shutil.rmtree(temp_dir)
                    print(f"  Removed: {temp_dir}")
        else:
            print(f"\nTemporary directories kept for debugging:")
            for temp_dir in temp_dirs:
                if temp_dir:
                    print(f"  {temp_dir}")


if __name__ == "__main__":
    main()
