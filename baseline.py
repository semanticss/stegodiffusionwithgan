

def gen_img_no_modification(prompt=prompt):
    latents = torch.randn(1, 4, 64, 64).to("cuda")  # Initial latent noise

    print(f"🔹 Generating Image Without Modification - Latents mean={latents.mean().item()}, std={latents.std().item()}")

    with torch.no_grad():
        stego_img = pipe(prompt=prompt, latents=latents).images[0]

    return stego_img
