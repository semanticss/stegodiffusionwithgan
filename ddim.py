import torch
import torchvision.transforms as transforms
from torchvision import datasets

# Importing torch libraries.

transform = transforms.Compose([ # .Compose allows you to perform multiple transformations at the same time.

    transforms.Resize((64, 64)), # Resizes images to 64 x 64 pixels. Can be changed (I think).
    transforms.ToTensor(), # Converts images to PyTorch tensor format.
    transforms.Normalize(0.5, 0.5) # Normalizes pixel values to a [-1, 1] range.

])

dataset = datasets.CelebA(root = r"C:\Users\huds0\Desktop\research\celeba", download = True, transform = transform) # Loads the CelebA dataset.
dataloader = torch.utils.data.DataLoader(dataset, batch_size = 64, shuffle = True) # Creates the DataLoader

