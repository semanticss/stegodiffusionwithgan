import cv2
import torch
import numpy as np
from diffusers import StableDiffusionPipeline
import torch.nn.functional as F
import matplotlib.pyplot as plt


def dummy_safety_checker(images, clip_input):
     return images, [False] * len(images)


pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda")
pipe.safety_checker = dummy_safety_checker  # Override filter

alpha = 0.35
threshold = 0.05
prompt = "a very high quality image"

def encode_latents(image):
    """ Encode an image into Stable Diffusion's latent space """
    image = image.unsqueeze(0).to("cuda")  # Ensure batch dimension
    latents = pipe.vae.encode(image).latent_dist.sample()
    return latents

def decode_latents(latents):
    """ Decode latents back to an image """
    image = pipe.vae.decode(latents).sample
    return image.squeeze(0).cpu()  # Remove batch dimension

def dct2(b):
    return cv2.dct(np.float32(b))

def invdct2(b):
    return cv2.idct(b)

def embed_message(latents, message, alpha = alpha):

    latents_np = latents.cpu().numpy()
    message_np = message.cpu().numpy()

    print(f"Initial Latents: mean={latents.mean().item()}, std={latents.std().item()}")


    latents_np = np.clip(latents_np, -1, 1)

    for i in range(4):
        latents_np[0, i, :, :] = dct2( latents_np[0, i, :, :])

        latents_np[0, i, :, :] = np.clip(latents_np[0, i, :, :], -1, 1)

        print(f"After DCT - Latent {i}: mean={latents_np[0, i].mean()}, std={latents_np[0, i].std()}")  # Debug step 2
        latents_np[0, i, -8:, -8:] += alpha * message_np[0, i, -8:, -8:] * latents_np[0, i, -8:, -8:]
        print(f"After Message Embedding - Latent {i}: mean={latents_np[0, i].mean()}, std={latents_np[0, i].std()}")  # Debug step 3
        latents_np[0, i, :, :] += np.random.uniform(-1e-6, 1e-6, size=latents_np[0, i, :, :].shape)
        print(f"After IDCT - Latent {i}: mean={latents_np[0, i].mean()}, std={latents_np[0, i].std()}")  # Debug step 4

        latents_modded = torch.tensor(latents_np, dtype=torch.float32).to(latents.device)

        if np.isnan(latents_np[0, i]).any():
            print(f"NaN detected after IDCT in channel {i}, replacing with zeros.")
            latents_np[0, i] = np.nan_to_num(latents_np[0, i])  # Replace NaNs with 0

        latents_modded = (latents_modded - latents_modded.mean()) / (latents_modded.std() + 1e-8)

        std = latents_modded.std()
        if std < 1e-6:  # If std is too small, prevent division by tiny value
            std = 1.0

        latents_modded = latents_modded / std

    return torch.tensor(latents_np, dtype=torch.float32).to(latents.device)

def get_message(latents, threshold = threshold):

    latents_np = latents.cpu().numpy()
    extracted_message = np.zeros_like(latents_np)

    for i in range(4):
        dct_coeffs = dct2(latents_np[0, i, :, :])
        extracted_message[0, i, -8:, -8:] = (dct_coeffs[-8:, -8:] > threshold).astype(int)

    return extracted_message

def gen_img(message, prompt = prompt, alpha = alpha):

    latents = torch.randn(1, 4, 64, 64).to('cuda')
    latents_w_message = embed_message(latents, message, alpha)

    with torch.no_grad():
        stego_img = pipe(prompt = prompt, latents = latents_w_message).images[0]

    return stego_img, latents_w_message

def get_message_from_stego(stego_img, theshold = threshold):

    image_tensor = torch.tensor(np.array(stego_img).astype(np.float32) / 127.5 - 1).permute(2, 0, 1).to("cuda")
    image_latents = encode_latents(image_tensor)
    extracted_message = get_message(image_latents, threshold)

    return extracted_message


message = torch.randint(0, 2, (1, 4, 64, 64)).float().to("cuda")

stego_image, latents_with_message = gen_img(message)

print(f"Latents mean: {latents_with_message.mean().item()}, std: {latents_with_message.std().item()}")

plt.imshow(stego_image)
plt.axis("off")
plt.show()

