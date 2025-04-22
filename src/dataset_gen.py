import os
import csv
from tqdm import tqdm
from random import choice
import torch
from pipeline import generate_stego_image, text_to_bit_tensor, bit_tensor_to_text, prepare_image_tensor, encode_latents, dct_extract

prompts = [
    "a macro photo of colorful leaves on stone",
    "a high-res photo of stitched fabric with geometric patterns",
    "a zoomed-in image of cracked blue ceramic tiles",
    "a close-up of glowing fiber optic wires",
    "a detailed shot of coral reef textures",
    "a macro shot of iridescent soap bubbles on glass",
    "a close-up of a rusted, peeling painted door",
    "an aerial view of colorful city rooftops",
    "a zoomed-in image of a circuit board with multicolored wires",
    "a macro photo of spilled glitter on black fabric",
    "a close-up of honeycomb with dripping honey",
    "a high-resolution image of woven baskets with different patterns",
    "a macro view of cracked lava rock",
    "a zoom-in of frosted window patterns",
    "a close-up of a jellyfish with glowing tentacles",
    "a detailed macro of frost on leaves",
    "a high-res scan of antique fabric embroidery",
    "a macro photo of layered paint chipping on wood",
    "a close-up of soap film with rainbow reflections",
    "a zoomed-in view of tree bark with moss and fungus",
    "a macro photo of flower petals with dew drops",
    "a highly detailed close-up of gemstone surfaces",
    "a macro image of organic cotton weave",
    "a close-up of rough sandpaper texture",
    "a detailed scan of reptile skin",
    "a close-up of an oil painting with visible brush strokes",
    "a macro photo of beetle shell with iridescent colors",
    "a highly magnified image of pollen grains",
    "a close-up of shattered stained glass",
    "a macro image of a sliced kiwi fruit",
    "a zoom-in on metallic car paint under sunlight",
    "a macro shot of a starfish texture",
    "a close-up of knitted wool with strong fiber contrast",
    "a high-res image of papyrus scroll details",
    "a detailed macro of sculpted ceramic glaze",
    "a close-up of aged parchment with ink",
    "a macro photo of fish scales under water",
    "a zoomed-in view of peeling wall posters",
    "a close-up of wax seals on old letters",
    "a macro image of frozen bubbles in ice",
    "a high-resolution scan of handwoven carpet",
    "a close-up of seashell ridges and texture",
    "a macro image of crystals growing on rock",
    "a close-up of LED screen pixels",
    "a detailed image of an old mosaic tile floor",
    "a macro view of tangled headphone wires",
    "a close-up of leather with natural wrinkles",
    "a macro photo of rusted chains on a dock",
    "a detailed view of braided horsehair ropes",
    "a high-res scan of decaying plant matter"
]

# Sample messages
messages = [
    "alpha2024", "diffusion42", "steganography", "GANlab",
    "latenttruth", "deepcode", "visualbytes", "messageME",
    "camouflage", "unseenword", "hiddenspace", "resilience",
    "encryptME", "texturemap", "smoothwave", "signalX"
]

# Output folders
base_path = "datasets/gan"
os.makedirs(base_path, exist_ok=True)
for split in ["train", "val"]:
    os.makedirs(f"{base_path}/{split}/clean", exist_ok=True)
    os.makedirs(f"{base_path}/{split}/stego", exist_ok=True)

csv_path = f"{base_path}/metadata.csv"
with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["index", "split", "prompt", "message", "recovered", "accuracy"])


    start_index = 400
    for j, prompt in enumerate(tqdm(prompts * 30)):
        i = start_index + j
        split = "train" if i % 5 != 0 else "val"
        msg = choice(messages)
        message_tensor = text_to_bit_tensor(msg)

        clean_img = generate_stego_image(prompt, message_tensor, alpha=0.0)
        clean_img.save(f"{base_path}/{split}/clean/{i:04}.png")

        stego_img = generate_stego_image(prompt, message_tensor, alpha=3.0)
        stego_img.save(f"{base_path}/{split}/stego/{i:04}.png")

        tensor = prepare_image_tensor(stego_img)
        latent = encode_latents(tensor)
        extracted = dct_extract(latent)
        recovered = bit_tensor_to_text(torch.tensor(extracted))
        acc = 100 * (message_tensor.cpu().numpy().round() == extracted.round().astype(int)).mean()

        writer.writerow([f"{i:04}.png", split, prompt, msg, recovered, f"{acc:.2f}"])

print("✅ Clean/stego dataset generation with tracking complete.")
