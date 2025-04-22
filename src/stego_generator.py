import torch
import torch.nn as nn

class StegoGenerator(nn.Module):
    def __init__(self, dct_shape=(16, 16), learnable_patch=True, learnable_alpha=False):
        super().__init__()
        self.learnable_patch = learnable_patch
        self.learnable_alpha = learnable_alpha

        if learnable_patch:
            self.dct_patch = nn.Parameter(torch.randn(1, 1, *dct_shape))  # Learnable 16x16 patch
        else:
            self.dct_patch = None  # Expects message input at runtime

        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(3.0))
        else:
            self.alpha = 3.0

    def forward(self, latents, message=None):
        latents = latents.clone()
        B, C, H, W = latents.shape
        device = latents.device

        patch = self.dct_patch.expand(B, 1, 16, 16) if self.learnable_patch else message
        alpha = self.alpha if self.learnable_alpha else torch.tensor(self.alpha, device=device)

        for i in [0]:  # Embed into channel 0
            for b in range(B):
                dct = torch.fft.fft2(latents[b, i])
                dct[16:32, 16:32] += alpha * patch[b, 0]
                idct = torch.fft.ifft2(dct).real
                latents[b, i] = idct

        latents = (latents - latents.mean()) / (latents.std() + 1e-8)
        return latents * 0.18215
