import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
import os

# Load the model
model = torch.jit.load(r'model.pth').eval()

# Load the background image
background_path = r'C:\Users\dgn\Downloads\background.jpg'
background = Image.open(background_path).convert('RGB')

# Directory containing the source images
source_folder = r'I:\Werkstudenten\Deepak_Raj\DATASETS\Private\Original_frames\PETS2001dataset1camera2'
output_folder = r'C:\Users\dgn\Desktop\Matting_V2\Output'

# Resize background image
background = background.resize((380, 244))

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Iterate through the images in the source folder
for filename in os.listdir(source_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
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
