import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
import os
import cv2
import numpy as np
import time

# Load the model
model = torch.jit.load(r'model.pth').eval()

# Directory containing the source images
source_folder = r'I:\Werkstudenten\Deepak_Raj\DATASETS\Private\Original_frames\SiemensGehen20m'
output_folder = r'C:\Users\dgn\Desktop\Matting_V2\Output'

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

def calculate_background(images_folder, num_frames=10):
    frame_buffer = []
    for filename in sorted(os.listdir(images_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Load image
            image_path = os.path.join(images_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue
            frame_buffer.append(image)
            if len(frame_buffer) >= num_frames:
                break
    
    if not frame_buffer:
        raise ValueError("No frames captured from the folder")

    # Calculate the average background from the last num_frames frames
    AllFrames = np.zeros_like(frame_buffer[0], dtype=float)
    for frame in frame_buffer:
        AllFrames += frame.astype(float)
    AllFrames /= len(frame_buffer)
    background = AllFrames.astype(np.uint8)
    
    # Convert to PIL image
    background_pil = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    return background_pil

# Calculate the background from the images folder
background = calculate_background(source_folder, num_frames=100).convert('RGB')

# Resize background image
background = background.resize((380, 244))

background.show()

# Track total processing time
total_start_time = time.time()

# Iterate through the images in the source folder
for filename in os.listdir(source_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        # Start timing for this frame
        frame_start_time = time.time()
        
        # Load source image and resize it
        source_path = os.path.join(source_folder, filename)
        source = Image.open(source_path).convert('RGB')
        source = source.resize((380, 244))

        # Convert images to tensors
        source_tensor = to_tensor(source).unsqueeze(0)
        background_tensor = to_tensor(background).unsqueeze(0)

        # Adjust model parameters based on image size
        if source_tensor.size(2) <= 2048 and source_tensor.size(3) <= 2048:
            model.backbone_scale = 1 / 4
            model.refine_sample_pixels = 80_000
        else:
            model.backbone_scale = 1 / 8
            model.refine_sample_pixels = 320_000

        # Process the images
        with torch.no_grad():
            alpha_matte, foreground = model(source_tensor, background_tensor)[:2]

        # Threshold the alpha matte to obtain a binary mask
        binary_mask = (alpha_matte > 0.05).float()

        # Save the output binary mask
        source_filename = os.path.splitext(filename)[0]
        output_path = os.path.join(output_folder, f'{source_filename}_binary_mask.png')
        to_pil_image(binary_mask[0].cpu()).save(output_path)

        # Print the path where the output binary mask is saved
        print(f'Output binary mask saved to: {output_path}')
        
        # Print the time taken for this frame
        frame_end_time = time.time()
        frame_time = frame_end_time - frame_start_time
        print(f'Time taken for {filename}: {frame_time:.4f} seconds')

# Print the total time taken for all frames
total_end_time = time.time()
total_time = total_end_time - total_start_time
print(f'Total time taken for all frames: {total_time:.4f} seconds')
