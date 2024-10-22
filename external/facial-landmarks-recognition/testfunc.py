import dlib
import cv2
import numpy as np

import face_alignment
from skimage import io
import torch
import os
import cv2
import numpy as np
import shutil
import csv 

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, dtype=torch.bfloat16, device='cuda')


class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def align_faces(original, render):
    original_landmarks = fa.get_landmarks(original)
    render_landmarks = fa.get_landmarks(render)
    if original_landmarks == None or render_landmarks == None:
        return []

    # Get landmarks for both images
    #original_landmarks = get_landmarks(original)
    #render_landmarks = get_landmarks(render)

    # Calculate affine transform
    transform = cv2.estimateAffinePartial2D(render_landmarks[0], original_landmarks[0])[0]

    # Apply affine transform to render
    rows, cols = original.shape[:2]
    aligned_render = cv2.warpAffine(render, transform, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=[0, 0, 0])

    # Create a mask for the aligned render
    gray = cv2.cvtColor(aligned_render, cv2.COLOR_BGR2GRAY)
    mask = (gray > 0).astype(np.uint8) * 255

    # Combine the aligned render with a black background
    black_bg = np.zeros_like(original)
    result = cv2.bitwise_and(aligned_render, aligned_render, mask=mask)
    result = cv2.add(result, black_bg)

    return result


def find_matching_original(original_folder, render_file):
    base_name = os.path.splitext(render_file)[0][:-2]
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        potential_file = base_name + ext
        print(potential_file)
        if os.path.exists(os.path.join(original_folder, potential_file)):
            return potential_file
    return None

import os
import cv2
import numpy as np

def process_image_folders_v1(original_folder, render_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process subfolders
    for root, dirs, files in os.walk(render_folder):
        for render_file in files:
            if render_file.lower().endswith('.png'):
                # Get relative path from render_folder
                rel_path = os.path.relpath(root, render_folder)

                # Construct paths
                render_path = os.path.join(root, render_file)
                original_subfolder = os.path.join(original_folder, rel_path)
                output_subfolder = os.path.join(output_folder, rel_path)

                # Create output subfolder if it doesn't exist
                os.makedirs(output_subfolder, exist_ok=True)

                output_path = os.path.join(output_subfolder, f"combined_{render_file}")

                # Find matching original file
                original_file = find_matching_original(original_subfolder, render_file)
                if not original_file:
                    print(f"Original not found for {render_file} in {original_subfolder}, skipping...")
                    continue

                original_path = os.path.join(original_subfolder, original_file)

                # Read images
                original = cv2.imread(original_path)
                render = cv2.imread(render_path)

                # Align faces
                try:
                    aligned_render = align_faces(original, render)
                except (TooManyFaces, NoFaces):
                    print(f"Face alignment failed for {original_path}, skipping...")
                    continue

                # Get landmarks
                try:
                    original_landmarks = get_landmarks(original)
                    aligned_render_landmarks = get_landmarks(aligned_render)
                except (TooManyFaces, NoFaces):
                    print(f"Landmark detection failed for {original_path}, skipping...")
                    continue

                # Annotate landmarks
                annotated_original = annotate_landmarks(original, original_landmarks)
                annotated_aligned_render = annotate_landmarks(aligned_render, aligned_render_landmarks)

                # Combine images side by side
                combined_image = np.hstack((annotated_original, annotated_aligned_render))

                # Save the combined result
                cv2.imwrite(output_path, combined_image)

                print(f"Processed {output_path}")


def process_image_folders(original_folder, render_folder, output_folder,reject_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(reject_folder, exist_ok=True)
    reject_geometry = reject_folder+'/geometry_detail/'
    reject_original = reject_folder+'/original/'
    os.makedirs(reject_geometry, exist_ok=True)
    os.makedirs(reject_original, exist_ok=True)
        # Create CSV file for rejected files
    csv_path = os.path.join(reject_folder, 'rejected_files.csv')
    rejected_files = []


    # Process subfolders
    for root, dirs, files in os.walk(render_folder):
        for render_file in files:
            if render_file.lower().endswith('.png'):
                # Get relative path from render_folder
                rel_path = os.path.relpath(root, render_folder)

                # Construct paths
                render_path = os.path.join(root, render_file)
                original_subfolder = os.path.join(original_folder, rel_path)
                output_subfolder = os.path.join(output_folder, rel_path)

                # Create output subfolder if it doesn't exist
                os.makedirs(output_subfolder, exist_ok=True)

                output_path = os.path.join(output_subfolder, f"resizedEmoca_{render_file}")

                # Find matching original file
                original_file = find_matching_original(original_subfolder, render_file)
                if not original_file:
                    print(f"Original not found for {render_file} in {original_subfolder}, skipping...")
                    continue

                original_path = os.path.join(original_subfolder, original_file)

                # Read images
                original = cv2.imread(original_path)
                render = cv2.imread(render_path)

                # Align faces
                try:
                    aligned_render = align_faces(original, render)
                    if aligned_render is []:
                        # Add the original file path to the rejected_files list
                        rejected_files.append(original_path)
                        print(f"Alignment failed for {render_file}. Added to rejected files list.")           
                        continue


                except (TooManyFaces, NoFaces):
                    print(f"Face alignment failed for {original_path}, skipping...")
                    continue


                # Annotate landmarks

                # Combine images side by side
                try:
                    annotate_landmarks_original = get_landmarks(original)
                    annotate_landmarks_aligned_render = get_landmarks(aligned_render)
                    #aligned_render = annotate_landmarks(aligned_render, annotate_landmarks_aligned_render)
                    original = annotate_landmarks(original, annotate_landmarks_original)
                    combined_image = np.hstack((original,aligned_render))
                    right_half = combined_image[:, combined_image.shape[1]//2:]


                    # Save the combined result
                    cv2.imwrite(output_path, right_half)
                                            # Add the original file path to the rejected_files list
                    print(f"Processed {output_path}")
                except Exception as e:
                    print(f"SKIPPED:: {output_path}")
                    rejected_files.append(original_path)

                    print(e)
        # Write rejected files to CSV
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Original File Path'])  # Header
        for file_path in rejected_files:
            csv_writer.writerow([file_path])

    print(f"Rejected files list saved to {csv_path}")


# Usage example
process_image_folders('/fast/vfourel/FaceGPT/Data/StableFaceData/affectnet_41k_AffectOnly/Manually_Annotated/Manually_Annotated_Images',
 '/fast/vfourel/FaceGPT/Data/StableFaceData/AffectNet41k_FlameRender_Descriptions_Images/affectnet_41k_AffectOnly/EmocaProcessed_38k/geometry_detail',
  '/fast/vfourel/FaceGPT/Data/StableFaceData/AffectNet41k_FlameRender_Descriptions_Images/affectnet_41k_AffectOnly/FLAMEResized/',
  '/fast/vfourel/FaceGPT/Data/StableFaceData/AffectNet41k_FlameRender_Descriptions_Images/affectnet_41k_AffectOnly/reject/')
