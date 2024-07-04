import os
from PIL import Image

def convert_png_to_jpg(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith(".png"):
            png_path = os.path.join(directory, filename)
            jpg_path = os.path.join(directory, os.path.splitext(filename)[0] + ".jpg")
            
            with Image.open(png_path) as img:
                rgb_img = img.convert("RGB")
                rgb_img.save(jpg_path, "JPEG")
            
            os.remove(png_path)
            print(f"Converted {png_path} to {jpg_path} and removed the original .png file.")

# Set the directory containing .png files
directory = r'I:\Werkstudenten\Deepak_Raj\DATASETS\Public\Public\Bulb-illumination'

convert_png_to_jpg(directory)
