import os
import re
import torch
from diffusers import StableDiffusionPipeline


MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "outputs"


def create_filename(prompt):
    filename = re.sub(r"[^a-zA-Z0-9]+", "_", prompt)
    filename = filename.strip("_").lower()

    return f"{filename[:80]}.png"


def generate_image(prompt):
    print("Loading Stable Diffusion model...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )

    pipe = pipe.to(device)

    print(f"Generating image on {device}...")

    image = pipe(prompt).images[0]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    filename = create_filename(prompt)
    output_file = os.path.join(OUTPUT_DIR, filename)

    image.save(output_file)

    print(f"Image saved successfully as: {output_file}")


if __name__ == "__main__":
    prompt = input("Enter your text prompt: ")
    generate_image(prompt)
