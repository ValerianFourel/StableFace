import csv
import os
import cv2
import numpy as np
import torch
from face_alignment import FaceAlignment, LandmarksType

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize the face alignment model with CUDA
fa = FaceAlignment(LandmarksType.TWO_D, flip_input=False, device=device)

def process_image(image_path):
    # Read the image
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Unable to read image {image_path}")
        return None

    # Get face landmarks
    landmarks = fa.get_landmarks(img)

    if landmarks is None or len(landmarks) == 0:
        print(f"No face detected in {image_path}")
        return None

    # Create a black background of the same size as the original image
    result = np.zeros(img.shape, dtype=np.uint8)

    try:
        # Get the first (and usually only) detected face
        landmarks = landmarks[0]
        landmarks_int = landmarks.astype(np.int32)

        # Define the indices for different facial features
        jaw = list(range(0, 17))
        left_eyebrow = list(range(17, 22))
        right_eyebrow = list(range(22, 27))
        nose = list(range(27, 36))
        left_eye = list(range(36, 42))
        right_eye = list(range(42, 48))
        mouth = list(range(48, 68))

        # Function to draw a line for a set of points
        def draw_line(points):
            for i in range(len(points) - 1):
                pt1 = tuple(landmarks_int[points[i]])
                pt2 = tuple(landmarks_int[points[i + 1]])
                cv2.line(result, pt1, pt2, (255, 255, 255), 1)

        # Draw lines for each facial feature
        draw_line(jaw)
        draw_line(left_eyebrow)
        draw_line(right_eyebrow)
        draw_line(nose)
        draw_line(left_eye + [left_eye[0]])  # Connect last point to first for eyes
        draw_line(right_eye + [right_eye[0]])
        draw_line(mouth + [mouth[0]])  # Connect last point to first for mouth

        # Add line between points 31 and 36
        cv2.line(result, tuple(landmarks_int[30]), tuple(landmarks_int[35]), (255, 255, 255), 1)

        return result
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

# Path to your CSV file
csv_file_path = '/fast/vfourel/FaceGPT/Data/StableFaceData/AffectNet41k_FlameRender_Descriptions_Images/processed_affectnet_paths.csv'

# Read the CSV file and process images
with open(csv_file_path, 'r') as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for row in csv_reader:
        if row['In FLAMEResized'] == '1':
            original_path = row['Original Path']

            # Process the image
            result_image = process_image(original_path)

            if result_image is not None:
                # Get the last subfolder and filename
                path_parts = original_path.split(os.sep)
                last_subfolder = path_parts[-2]
                filename = os.path.splitext(path_parts[-1])[0]

                # Create the output directory if it doesn't exist
                output_dir = os.path.join('aligned_images', last_subfolder)
                os.makedirs(output_dir, exist_ok=True)

                # Create the output path with .png extension
                output_path = os.path.join(output_dir, f"{filename}.png")

                # Save the processed image
                cv2.imwrite(output_path, result_image)
                print(f"Processed and saved: {output_path}")
