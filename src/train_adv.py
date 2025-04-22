import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from stego_generator import StegoGenerator
from pipeline import encode_latents, prepare_image_tensor, dct_extract, text_to_bit_tensor
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
).to("cuda")
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)
    
def generate_latents(prompt, batch_size):
    base_images = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=12.5, num_images_per_prompt=batch_size).images
    tensors = torch.cat([prepare_image_tensor(img) for img in base_images], dim=0)
    return encode_latents(tensors)

import matplotlib.pyplot as plt

def train():
    device = torch.device("cuda")
    G = StegoGenerator().to(device)
    D = Discriminator().to(device)
    optimizer_G = optim.Adam(G.parameters(), lr=1e-4)
    optimizer_D = optim.Adam(D.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    batch_size = 2
    message = text_to_bit_tensor("secret123", shape=(1, 4, 16, 16)).expand(batch_size, -1, -1, -1)
    prompt = "a macro shot of coral reef textures"

    losses_D, losses_G = [], []
    for epoch in range(100):
        latents = generate_latents(prompt, batch_size).to(device)
        clean_images = pipe.vae.decode(latents / 0.18215).sample
        stego_latents = G(latents.detach(), message=message.detach())
        stego_images = pipe.vae.decode(stego_latents / 0.18215).sample

        real_labels = torch.zeros(batch_size, 1, device=device)
        fake_labels = torch.ones(batch_size, 1, device=device)

        D_real = D((clean_images / 2 + 0.5).clamp(0, 1).to(torch.float32))
        D_fake = D((stego_images / 2 + 0.5).clamp(0, 1).detach().to(torch.float32))

        loss_D = criterion(D_real, real_labels) + criterion(D_fake, fake_labels)
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        D_fake_for_G = D((stego_images / 2 + 0.5).clamp(0, 1).to(torch.float32))
        loss_G = criterion(D_fake_for_G, real_labels)

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    
        import gc
        torch.cuda.empty_cache()
        gc.collect()

        print(f"Epoch {epoch+1}: Loss_D = {loss_D.item():.4f}, Loss_G = {loss_G.item():.4f}")
        losses_D.append(loss_D.item())
        losses_G.append(loss_G.item())

    os.makedirs("results", exist_ok=True)
    plt.plot(losses_D, label="Discriminator Loss")
    plt.plot(losses_G, label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training Loss Curves")
    plt.legend()
    plt.savefig("results/gan_loss_curves.png")
    plt.close()

if __name__ == "__main__":
    train()
