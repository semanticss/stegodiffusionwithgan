# DCT Steganography with Stable Diffusion

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionPipeline
import pandas as pd

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype = torch.float16).to("cuda")
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

MSG_SIZE = 16
EMBED_REGION = slice(16, 32)
DEFAULT_ALPHA = 3.0
USE_TEXTURE_CHANNEL = 0
NUM_SAMPLES = 5

H_enc = np.array([[1,1,0,1], [1,0,1,1], [1,0,0,0], [0,1,1,1], [0,1,0,0], [0,0,1,0], [0,0,0,1]], dtype=int)
H_dec = np.array([[1,0,1,0,1,0,1], [0,1,1,0,0,1,1], [0,0,0,1,1,1,1]], dtype=int)

syndrome_table = {
    (0, 0, 0): 0,
    (0, 0, 1): 6,
    (0, 1, 0): 4,
    (0, 1, 1): 2,
    (1, 0, 0): 3,
    (1, 0, 1): 0,
    (1, 1, 0): 1,
    (1, 1, 1): 5
}

def hamming_encode(bits):
    padded = bits + [0] * ((4 - len(bits) % 4) % 4)
    chunks = [padded[i:i+4] for i in range(0, len(padded), 4)]
    encoded = [H_enc @ np.array(c) % 2 for c in chunks]
    return np.concatenate(encoded)

def hamming_decode(bits):
    chunks = [bits[i:i+7] for i in range(0, len(bits), 7)]
    decoded = []
    for c in chunks:
        c = np.array(c)
        c = c.reshape(-1, 1)
        if c.shape[0] != 7:
            continue
        s = tuple((H_dec @ c % 2).flatten())
        if s != (0,0,0):
            error_bit = syndrome_table.get(s, 0)
            if error_bit:
                c[error_bit-1] ^= 1
        decoded.append(c[2])  
        decoded.append(c[4])  
        decoded.append(c[5])  
        decoded.append(c[6])  
    return decoded

def text_to_bit_tensor(text, shape=(1, 4, 16, 16), device="cuda"):
    bits = ''.join(f"{ord(c):08b}" for c in text)
    encoded_bits = hamming_encode([int(b) for b in bits])
    required_size = np.prod(shape)
    if len(encoded_bits) < required_size:
        encoded_bits = np.pad(encoded_bits, (0, required_size - len(encoded_bits)), constant_values=0)
    else:
        encoded_bits = encoded_bits[:required_size]
    bit_array = np.array(encoded_bits, dtype=np.float32).reshape(shape)
    return torch.tensor(bit_array, device=device)

def bit_tensor_to_text(tensor):
    bits = tensor.detach().cpu().numpy().flatten().round().astype(int)
    decoded_bits = hamming_decode(bits.tolist())
    chars = []
    for i in range(0, len(decoded_bits) - 7, 8):
        byte_str = ''.join(str(int(b)) for b in decoded_bits[i:i+8])
        try:
            chars.append(chr(int(byte_str, 2)))
        except ValueError:
            chars.append('?')
    return ''.join(chars).rstrip('\x00')

def dct_embed(latents, message, alpha=DEFAULT_ALPHA):
    latents_np = latents.detach().cpu().float().numpy()
    message_np = message.detach().cpu().float().numpy()
    for i in [USE_TEXTURE_CHANNEL]:
        dct = cv2.dct(latents_np[0, i])
        dct[EMBED_REGION, EMBED_REGION] += alpha * message_np[0, i]
        latents_np[0, i] = cv2.idct(dct)
    latents_np = (latents_np - latents_np.mean()) / (latents_np.std() + 1e-8)
    return torch.tensor(latents_np, device=latents.device, dtype=latents.dtype) * 0.18215

def dct_extract(latents, threshold="mean"):
    latents_np = (latents.detach().cpu().float().numpy()) / 0.18215
    extracted = np.zeros((1, 4, MSG_SIZE, MSG_SIZE))
    for i in [USE_TEXTURE_CHANNEL]:
        dct = cv2.dct(latents_np[0, i])
        patch = dct[EMBED_REGION, EMBED_REGION]
        t = patch.mean() if threshold == "mean" else threshold
        extracted[0, i] = (patch > t).astype(int)
    return extracted

def prepare_image_tensor(image: Image.Image):
    image = np.array(image).astype(np.float32) / 255.0
    image = (image - 0.5) * 2.0
    return torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.float16)

def encode_latents(image_tensor):
    return pipe.vae.encode(image_tensor).latent_dist.sample() * 0.18215

def generate_stego_image(prompt, message, alpha=DEFAULT_ALPHA):
    generator = torch.manual_seed(42)
    base = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=12.5, generator=generator).images[0]
    image_tensor = prepare_image_tensor(base)
    latents = encode_latents(image_tensor)
    modified_latents = dct_embed(latents, message, alpha)
    decoded = pipe.vae.decode(modified_latents / 0.18215).sample
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    decoded = decoded.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
    return Image.fromarray((decoded * 255).astype(np.uint8))

text_message = "secret123"
message = text_to_bit_tensor(text_message)

prompts = [
    "a detailed macro photograph of multicolored pebbles on a beach",
    "a high-resolution close-up of rusted metal with peeling paint",
    "a close-up of moss and lichen growing on stone with rich texture",
    "a macro image of iridescent soap bubbles on a reflective surface",
    "a high-resolution macro of a circuit board with colorful resistors and wires"
]

recovered_votes = []
for i in range(NUM_SAMPLES):
    prompt = prompts[i % len(prompts)]
    stego_image = generate_stego_image(prompt, message)
    image_tensor = prepare_image_tensor(stego_image)
    latents = encode_latents(image_tensor)
    extracted = dct_extract(latents)
    recovered_votes.append(np.round(extracted).astype(int))

recovered_votes = np.array(recovered_votes)
bit_majority = (np.mean(recovered_votes, axis=0) > 0.5).astype(int)
recovered_text = bit_tensor_to_text(torch.tensor(bit_majority))
original_text = text_message
bit_accuracy = 100 * (bit_majority == message.cpu().numpy()).mean()

print(f"Original:  {original_text}")
print(f"Recovered: {recovered_text}")
print(f"Bit Accuracy: {bit_accuracy:.2f}%")

plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.imshow(stego_image)
plt.title("Stego Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(bit_majority[0, USE_TEXTURE_CHANNEL], cmap='gray')
plt.title("Recovered Bitplane")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow((bit_majority != message.cpu().numpy()).astype(float)[0, USE_TEXTURE_CHANNEL], cmap='hot')
plt.title("Error Heatmap")
plt.axis("off")

plt.tight_layout()
plt.show()

# Save images to datasets folder
import os
os.makedirs("datasets/stego", exist_ok=True)

stego_image.save("datasets/stego/stego_output.png")
Image.fromarray((bit_majority[0, USE_TEXTURE_CHANNEL] * 255).astype(np.uint8)).save("datasets/stego/recovered_bitplane.png")
Image.fromarray(((bit_majority != message.cpu().numpy()).astype(float)[0, USE_TEXTURE_CHANNEL] * 255).astype(np.uint8)).save("datasets/stego/error_map.png")

os.makedirs("results", exist_ok=True)
pd.DataFrame([{
    "message": text_message,
    "recovered_text": recovered_text,
    "bit_accuracy": bit_accuracy
}]).to_csv("results/metrics.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(pipe(prompt=prompts[0], generator=torch.manual_seed(42)).images[0])
axes[0].set_title("Original Prompt Output")
axes[0].axis("off")
axes[1].imshow(stego_image)
axes[1].set_title("Stego Image")
axes[1].axis("off")
plt.tight_layout()
plt.show()
