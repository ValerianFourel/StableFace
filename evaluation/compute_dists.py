#!/usr/bin/env python3
"""
Standalone DISTS Calculator for Nested Directory Structures

This script computes DISTS (Deep Image Structure and Texture Similarity) scores 
between directories that contain images in subfolders.

DISTS is computed pairwise between corresponding images and then averaged.

Usage:
    python compute_dists.py --original /path/to/original --baseline /path/to/baseline --finetuned /path/to/finetuned

Dependencies:
    pip install torch torchvision piq pillow
"""

import argparse
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
try:
    import piq
except ImportError:
    print("Error: Please install piq library: pip install piq")
    exit(1)


def find_all_images(directory: Path) -> List[Path]:
    """Find all image files in directory and subdirectories."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = []

    for file_path in directory.rglob('*'):
        if file_path.is_file() and file_path.suffix in image_extensions:
            image_files.append(file_path)

    return sorted(image_files)  # Sort for consistent ordering


def validate_image(image_path: Path) -> bool:
    """Check if image file is valid and can be opened."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        print(f"Invalid image {image_path}: {e}")
        return False


def load_and_preprocess_image(image_path: Path, size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
    """Load image and convert to tensor format expected by DISTS."""
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')

            # Resize to consistent size
            img = img.resize(size, Image.Resampling.LANCZOS)

            # Convert to tensor and normalize to [0, 1]
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

            tensor = transform(img)
            return tensor
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def create_image_pairs(original_dir: Path, comparison_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Create pairs of corresponding images between directories.
    Matches images based on their relative path structure.
    """
    original_images = find_all_images(original_dir)
    comparison_images = find_all_images(comparison_dir)

    # Create mapping based on relative paths
    original_map = {}
    for img_path in original_images:
        rel_path = img_path.relative_to(original_dir)
        # Remove extension and use stem for matching
        key = str(rel_path.with_suffix(''))
        original_map[key] = img_path

    comparison_map = {}
    for img_path in comparison_images:
        rel_path = img_path.relative_to(comparison_dir)
        key = str(rel_path.with_suffix(''))
        comparison_map[key] = img_path

    # Find matching pairs
    pairs = []
    matched_keys = set(original_map.keys()) & set(comparison_map.keys())

    for key in sorted(matched_keys):
        pairs.append((original_map[key], comparison_map[key]))

    unmatched_original = set(original_map.keys()) - matched_keys
    unmatched_comparison = set(comparison_map.keys()) - matched_keys

    if unmatched_original:
        print(f"Warning: {len(unmatched_original)} images in original directory have no match")
        if len(unmatched_original) <= 5:
            for key in list(unmatched_original)[:5]:
                print(f"  Unmatched original: {key}")

    if unmatched_comparison:
        print(f"Warning: {len(unmatched_comparison)} images in comparison directory have no match")
        if len(unmatched_comparison) <= 5:
            for key in list(unmatched_comparison)[:5]:
                print(f"  Unmatched comparison: {key}")

    return pairs


def compute_dists_batch(original_images: List[torch.Tensor], 
                       comparison_images: List[torch.Tensor], 
                       device: str = "cuda",
                       batch_size: int = 16) -> float:
    """Compute DISTS scores in batches to manage memory."""

    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"
        print("CUDA not available, using CPU")

    # Initialize DISTS metric
    dists_metric = piq.DISTS(reduction='none').to(device)

    all_scores = []

    # Process in batches
    for i in range(0, len(original_images), batch_size):
        batch_end = min(i + batch_size, len(original_images))

        # Prepare batch
        orig_batch = torch.stack(original_images[i:batch_end]).to(device)
        comp_batch = torch.stack(comparison_images[i:batch_end]).to(device)

        try:
            with torch.no_grad():
                # Compute DISTS scores for the batch
                batch_scores = dists_metric(orig_batch, comp_batch)
                all_scores.extend(batch_scores.cpu().numpy().tolist())

            print(f"  Processed batch {i//batch_size + 1}/{(len(original_images) + batch_size - 1)//batch_size}")

        except Exception as e:
            print(f"Error processing batch starting at {i}: {e}")
            # Try individual images in this batch
            for j in range(i, batch_end):
                try:
                    with torch.no_grad():
                        single_orig = original_images[j].unsqueeze(0).to(device)
                        single_comp = comparison_images[j].unsqueeze(0).to(device)
                        score = dists_metric(single_orig, single_comp)
                        all_scores.append(score.cpu().item())
                except Exception as e2:
                    print(f"Error processing individual image {j}: {e2}")
                    all_scores.append(float('inf'))

    if not all_scores:
        return float('inf')

    # Filter out infinite values
    valid_scores = [s for s in all_scores if s != float('inf')]
    if not valid_scores:
        return float('inf')

    return sum(valid_scores) / len(valid_scores)


def compute_dists_robust(original_dir: Path, comparison_dir: Path, device: str = "cuda") -> Tuple[float, int]:
    """
    Compute DISTS with robust error handling.
    Returns (average_dists_score, number_of_valid_pairs)
    """
    print(f"Creating image pairs between {original_dir.name} and {comparison_dir.name}...")
    pairs = create_image_pairs(original_dir, comparison_dir)

    if len(pairs) < 1:
        print("Error: No matching image pairs found!")
        return float('inf'), 0

    print(f"Found {len(pairs)} matching image pairs")

    # Load and preprocess images
    print("Loading and preprocessing images...")
    original_tensors = []
    comparison_tensors = []
    valid_pairs = 0

    for i, (orig_path, comp_path) in enumerate(pairs):
        if (i + 1) % 50 == 0:
            print(f"  Loaded {i + 1}/{len(pairs)} pairs")

        orig_tensor = load_and_preprocess_image(orig_path)
        comp_tensor = load_and_preprocess_image(comp_path)

        if orig_tensor is not None and comp_tensor is not None:
            original_tensors.append(orig_tensor)
            comparison_tensors.append(comp_tensor)
            valid_pairs += 1
        else:
            print(f"Skipping invalid pair: {orig_path.name} <-> {comp_path.name}")

    if valid_pairs < 1:
        print("Error: No valid image pairs after preprocessing!")
        return float('inf'), 0

    print(f"Successfully loaded {valid_pairs} valid pairs")

    # Compute DISTS scores
    print("Computing DISTS scores...")

    # Try different batch sizes if memory issues occur
    batch_sizes = [32, 16, 8, 4, 1]

    for batch_size in batch_sizes:
        try:
            print(f"Attempting DISTS calculation with batch_size={batch_size}")
            avg_dists = compute_dists_batch(original_tensors, comparison_tensors, device, batch_size)
            if avg_dists != float('inf'):
                print(f"✓ DISTS calculation successful with batch_size={batch_size}")
                return avg_dists, valid_pairs
            else:
                print(f"✗ DISTS calculation returned invalid result with batch_size={batch_size}")
        except Exception as e:
            print(f"✗ Failed with batch_size={batch_size}: {e}")
            continue

    print("All batch sizes failed!")
    return float('inf'), valid_pairs


def main():
    parser = argparse.ArgumentParser(description="Compute DISTS scores for nested directory structures")
    parser.add_argument("--original", required=True, help="Path to original images directory")
    parser.add_argument("--baseline", required=True, help="Path to baseline generated images directory") 
    parser.add_argument("--finetuned", required=True, help="Path to finetuned generated images directory")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to use for computation")
    parser.add_argument("--image-size", default=256, type=int, help="Resize images to this size (default: 256)")

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
    print("DISTS COMPUTATION FOR NESTED DIRECTORIES")
    print("=" * 60)

    # Count images in each directory
    original_images = find_all_images(original_dir)
    baseline_images = find_all_images(baseline_dir)
    finetuned_images = find_all_images(finetuned_dir)

    print(f"Found {len(original_images)} images in original directory")
    print(f"Found {len(baseline_images)} images in baseline directory") 
    print(f"Found {len(finetuned_images)} images in finetuned directory")

    if len(original_images) < 1 or len(baseline_images) < 1 or len(finetuned_images) < 1:
        print("Error: Need at least 1 image in each directory for DISTS calculation")
        return

    # Compute DISTS scores
    print("\n" + "=" * 60)
    print("COMPUTING DISTS SCORES")
    print("=" * 60)

    print(f"\nImage preprocessing size: {args.image_size}x{args.image_size}")

    print("\n1. Computing DISTS: Original ↔ Baseline")
    dists_baseline, pairs_baseline = compute_dists_robust(original_dir, baseline_dir, args.device)

    print("\n2. Computing DISTS: Original ↔ Finetuned")
    dists_finetuned, pairs_finetuned = compute_dists_robust(original_dir, finetuned_dir, args.device)

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if dists_baseline != float('inf'):
        print(f"DISTS (Original ↔ Baseline):   {dists_baseline:8.6f} (over {pairs_baseline} pairs)")
    else:
        print(f"DISTS (Original ↔ Baseline):   FAILED TO COMPUTE")

    if dists_finetuned != float('inf'):
        print(f"DISTS (Original ↔ Finetuned):  {dists_finetuned:8.6f} (over {pairs_finetuned} pairs)")
    else:
        print(f"DISTS (Original ↔ Finetuned):  FAILED TO COMPUTE")

    print("\nNote: Lower DISTS values indicate better structural/textural similarity to original images")
    print("DISTS range: [0, 1] where 0 = perfect similarity, 1 = maximum dissimilarity")

    if dists_baseline != float('inf') and dists_finetuned != float('inf'):
        if dists_finetuned < dists_baseline:
            improvement = dists_baseline - dists_finetuned
            improvement_pct = (improvement / dists_baseline) * 100
            print(f"✓ Finetuned model is better by {improvement:.6f} DISTS points ({improvement_pct:.1f}% improvement)")
        elif dists_baseline < dists_finetuned:
            degradation = dists_finetuned - dists_baseline
            degradation_pct = (degradation / dists_baseline) * 100
            print(f"✗ Baseline model is better by {degradation:.6f} DISTS points ({degradation_pct:.1f}% worse)")
        else:
            print("→ Both models perform equally")


if __name__ == "__main__":
    main()

