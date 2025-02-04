
import cv2
import torch
import numpy as np
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("mps")




def dct2(block):
    """Apply 2D Discrete Cosine Transform (DCT)."""
    return cv2.dct(np.float32(block))

def idct2(block):
    """Apply 2D Inverse Discrete Cosine Transform (IDCT)."""
    return cv2.idct(block)


def embed_message_in_dct(image, message, alpha=0.1):
    """
    Encode a binary message into the high-frequency DCT coefficients of an image.
    Args:
        image: Input image (grayscale or single channel).
        message: Binary message as a NumPy array (same shape as image).
        alpha: Strength of embedding (higher values change the image more).
    Returns:
        Modified image with hidden message.
    """
    # Convert image to frequency domain
    dct_coeffs = dct2(image)

    # Modify high-frequency components to embed the message
    height, width = dct_coeffs.shape
    dct_coeffs[height//2:, width//2:] += alpha * message[height//2:, width//2:]

    # Convert back to spatial domain
    modified_image = idct2(dct_coeffs)

    return np.clip(modified_image, 0, 255)  # Ensure valid pixel range




def extract_message_from_dct(image, threshold=0.05):
    """
    Extract hidden binary message from the high-frequency DCT coefficients of an image.
    Args:
        image: Image with an embedded message.
        threshold: Decision threshold to recover the binary message.
    Returns:
        Extracted binary message.
    """
    # Convert image to frequency domain
    dct_coeffs = dct2(image)

    # Extract high-frequency coefficients
    height, width = dct_coeffs.shape
    extracted_message = np.zeros_like(dct_coeffs)
    extracted_message[height//2:, width//2:] = (dct_coeffs[height//2:, width//2:] > threshold).astype(int)

    return extracted_message


def mn_projection(message, noise): # Message to noise.
    projected_noise = np.abs(noise) * (2 * message - 1)
    return projected_noise

def mb_projection(message): # Message to binary.
    return 2 * message - 1

def mc_projection(message, noise): # Message to Centered Binary.
    sign = np.random.choice([-1, 1], size=message.shape)
    projected_noise = np.sqrt(2) * message * sign
    return projected_noise

def embed_message(message, noise, method="MN"):
    if method == "MN":
        return mn_projection(message, noise)
    elif method == "MB":
        return mb_projection(message)
    elif method == "MC":
        return mc_projection(message, noise)
    else:
        raise ValueError("Invalid projection method")
    

def generate_stego_image(message, model, method="DCT", alpha=0.1):
    """
    Generate a stego image by encoding a message into the frequency domain of noise.
    Args:
        message: Binary message in the shape of latent noise.
        model: Stable Diffusion pipeline.
        method: Encoding method (DCT or others).
        alpha: Strength of message embedding.
    Returns:
        Generated stego image.
    """
    # Generate initial Gaussian noise
    noise = np.random.randn(1,4, 64, 64).astype(np.float32) * 255  # Simulate latent noise as a 64x64 grayscale image

    # Apply DCT message encoding
    if method == "DCT":
        noise = embed_message_in_dct(noise, message, alpha)

    # Convert noise to tensor format
    latent_noise = torch.tensor(noise, dtype=torch.float32).squeeze().unsqueeze(0).to("mps")
    print(f"Latent Noise Shape: {latent_noise.shape}")

    with torch.no_grad():
        stego_image = model(prompt="A high-quality image", latents=latent_noise)

    return stego_image.images[0]



def extract_message_from_stego(stego_image):
    """
    Extract a hidden message from a generated stego image.
    """
    # Convert image to grayscale
    gray_image = np.array(stego_image.convert("L"))

    # Extract message using DCT
    extracted_message = extract_message_from_dct(gray_image)

    return extracted_message

    


message = np.random.randint(0, 2, (3, 64, 64))  # 3 channels, 64x64 image

# Generate Stego Image
stego_image = generate_stego_image(message, pipe, method="MB")

# Extract the message.
recovered_message = extract_message_from_stego(stego_image, pipe, method="MB")

# Accuracy checker.
accuracy = (message == recovered_message).mean()
print(f"Message Extraction Accuracy: {accuracy * 100:.2f}%")


message = np.random.randint(0, 2, (64, 64))

# Load Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("mps")

# Generate stego image
stego_image = generate_stego_image(message, pipe, method="DCT")

# Extract hidden message
recovered_message = extract_message_from_stego(stego_image)

# Compare accuracy
accuracy = (message == recovered_message).mean()
print(f"Message Extraction Accuracy: {accuracy * 100:.2f}%")

