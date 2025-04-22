import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader

# ==== Simple Discriminator for Steganalysis ====
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ==== Data Loading ====
def get_dataloader(data_path, batch_size):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(data_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ==== Training Loop ====
def train_gan(data_path, epochs=20, batch_size=16, lr=2e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    discriminator = Discriminator().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    dataloader = get_dataloader(data_path, batch_size)

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(dataloader):
            real = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)  # 0: clean, 1: stego

            preds = discriminator(real)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch {i}/{len(dataloader)} Loss: {loss.item():.4f}")

    torch.save(discriminator.state_dict(), "results/discriminator.pth")
    print("✅ Training complete. Model saved.")

if __name__ == "__main__":
    # Expects data in: datasets/gan/train/clean and datasets/gan/train/stego
    train_gan("datasets/gan/train", epochs=10, batch_size=16)
