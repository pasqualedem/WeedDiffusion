import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import os
from pathlib import Path


def generate_overlay(image, width_fraction=0.3, direction="right"):
    width, height = image.size
    overlay = Image.new("L", (width, height), 0)
    start_x = int(width * (1 - width_fraction)) if direction == "right" else 0
    end_x = width if direction == "right" else int(width_fraction * width)

    for x in range(start_x, end_x):
        for y in range(height):
            overlay.putpixel((x, y), 255)

    return overlay


def run_inpainting(pipe, image, overlay, prompt, out_path):
    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=overlay,
        guidance_scale=7.5,
        num_inference_steps=300,
        height=1024,
        width=1024,
    ).images[0]
    result.save(out_path)
    return result


def inpaint_crops(
    root="data/PhenoBench_aug/dreambooth_37", checkpoint="out/model_crops", input_images=None
):

    input_images = Path(input_images)
    prompt = "sks crop"
    output_dir = Path(root) / "inpainting_outputs"
    os.makedirs(output_dir, exist_ok=True)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        checkpoint, torch_dtype=torch.float16
    ).to("cuda")

    fractions = [0.2, 0.3, 0.4]
    directions = ["left", "right"]
    image_files = [
        f
        for f in os.listdir(input_images)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for img_name in image_files:
        img_path = os.path.join(input_images, img_name)
        image = Image.open(img_path).convert("RGB")
        base_name = os.path.splitext(img_name)[0]

        for direction in directions:
            for frac in fractions:
                overlay = generate_overlay(
                    image, width_fraction=frac, direction=direction
                )
                suffix = f"{direction}_{int(frac * 100)}"
                out_name = f"{base_name}_{suffix}.png"
                out_path = os.path.join(output_dir, out_name)

                print(f"Processing {img_name} | {suffix}")
                result = run_inpainting(pipe, image, overlay, prompt, out_path)
