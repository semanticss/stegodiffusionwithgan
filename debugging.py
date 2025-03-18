import torch
from PIL import Image
import os

print("Torch CUDA available:", torch.cuda.is_available())
print("Torch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)


# Define the path to an image inside 'celeba' folder
image_path = os.path.join("/celeba", "000001.jpg")  # Replace with an actual image name

# Open the image
try:
    img = Image.open("celeba\celeba\img_align_celeba\000001.jpg")
    print(f"Image Dimensions: {img.size}")  # (width, height)
except FileNotFoundError:
    print("Error: Image not found!")

