import os
from pathlib import Path
import wget

from diffusers import StableDiffusionPipeline
import torch

import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

import random
from skimage.transform import AffineTransform, warp
from PIL import Image


def generate_weeds(
    checkpoint="out/model_weeds",
    root="data/PhenoBench_aug/dreambooth_37",
    num_images=5,
):

    weeds_folder = Path(root) / "generated_weeds"
    os.makedirs(weeds_folder, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        checkpoint,
        torch_dtype=torch.float16,
    ).to("cuda")

    # Prompt used in the training
    prompt = "sks weed"

    images = pipe(
        prompt=prompt,
        num_inference_steps=300,
        guidance_scale=7.5,
        num_images_per_prompt=num_images,
        height=1024,
        width=1024,
    ).images

    # Show all images
    for i, img in enumerate(images):
        img.show(title=f"Weed {i+1}")

    # Save generated images
    for i, img in enumerate(images):
        img.save(weeds_folder / f"generated_weed_{i+1}.png")


def get_weed_and_mask(weed_path):
    # Load the weed image with alpha channel
    weed = cv2.imread(weed_path, cv2.IMREAD_UNCHANGED)
    if weed.shape[2] != 4:
        raise ValueError(f"{weed_path} does not have an alpha channel!")

    # Channel separation
    b, g, r, a = cv2.split(weed)
    mask = (a > 0).astype(np.uint8) * 255  # For saving as PNG (0 or 255)

    return weed, mask


def extract_weeds(root="data/PhenoBench_aug/dreambooth_37"):

    generated_weeds = Path(root) / "generated_weeds"
    segmented_weeds = Path(root) / "segmented_weeds"
    weed_masks = Path(root) / "segmented_weeds_masks"

    os.makedirs(segmented_weeds, exist_ok=True)
    os.makedirs(weed_masks, exist_ok=True)

    sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    if not os.path.exists(sam_checkpoint):
        wget.download(
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            sam_checkpoint,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    for image_path in os.listdir(generated_weeds):
        image_path = os.path.join(generated_weeds, image_path)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Masks generation
        masks = mask_generator.generate(image_rgb)
        print(f"Found {len(masks)} masks")

        # Filter masks to find green plant masks
        plant_masks = []
        for m in masks:
            seg = m["segmentation"]
            area = m["area"]

            # Mean color of the mask
            masked_pixels = image_rgb[seg]
            avg_color = masked_pixels.mean(axis=0)

            # Check if the mask is predominantly green
            if (
                avg_color[1] > 70
                and avg_color[1] > avg_color[0] * 0.9
                and avg_color[1] > avg_color[2] * 0.9
            ):
                plant_masks.append(m)

        # Use the largest mask
        plant_mask = max(plant_masks, key=lambda x: x["area"])["segmentation"]
        plant_mask = plant_mask.astype(np.uint8) * 255

        # Create PNG with transparency
        b, g, r = cv2.split(image)
        plant_png = cv2.merge((b, g, r, plant_mask))
        cv2.imwrite(
            os.path.join(segmented_weeds, os.path.basename(image_path)), plant_png
        )

    print(f"Cut out weeds saved to {segmented_weeds}")

    # Process each weed image
    for weed_file in os.listdir(segmented_weeds):
        weed_path = os.path.join(segmented_weeds, weed_file)
        try:
            weed, mask = get_weed_and_mask(weed_path)
            print(
                f"✅ Processed {weed_file} — shape: {weed.shape}, mask pixels: {np.sum(mask > 0)}"
            )

            # Save mask
            base_name = os.path.splitext(weed_file)[0]
            cv2.imwrite(os.path.join(weed_masks, f"{base_name}_mask.png"), mask)

        except ValueError as e:
            print(f"⚠️ {e}")


def random_geometric_transform(weed_img, weed_mask):
    transform = AffineTransform(
        scale=(random.uniform(0.8, 1.2), random.uniform(0.8, 1.2)),
        rotation=random.uniform(-0.2, 0.2),
        shear=random.uniform(-0.1, 0.1),
        translation=(random.randint(-5, 5), random.randint(-5, 5)),
    )

    weed_img = warp(weed_img, transform.inverse, preserve_range=True).astype(np.uint8)
    weed_mask = warp(weed_mask, transform.inverse, preserve_range=True, order=0).astype(
        np.uint8
    )

    return weed_img, weed_mask


def crop_to_content(weed_img, weed_mask):
    ys, xs = np.where(weed_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, None
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return (
        weed_img[y_min : y_max + 1, x_min : x_max + 1],
        weed_mask[y_min : y_max + 1, x_min : x_max + 1],
    )


def paste_weed(image, mask, weed_img, weed_mask, position):
    x, y = position
    h, w = weed_img.shape[:2]

    if y + h > image.shape[0] or x + w > image.shape[1]:
        return image, mask

    roi_img = image[y : y + h, x : x + w]
    roi_mask = mask[y : y + h, x : x + w]

    alpha = weed_img[:, :, 3] / 255.0
    for c in range(3):
        roi_img[:, :, c] = (1 - alpha) * roi_img[:, :, c] + alpha * weed_img[:, :, c]

    roi_mask[weed_mask > 0] = 2
    image[y : y + h, x : x + w] = roi_img
    mask[y : y + h, x : x + w] = roi_mask

    return image, mask


def find_background_spots(mask, weed_mask, num_trials=200):
    h_img, w_img = mask.shape
    h_weed, w_weed = weed_mask.shape

    for _ in range(num_trials):
        x = random.randint(0, w_img - w_weed)
        y = random.randint(0, h_img - h_weed)
        region = mask[y : y + h_weed, x : x + w_weed]
        if np.all(region == 0):
            return x, y
    return None


def augment_weed(weed_img, weed_mask):
    k = random.choice([0, 1, 2, 3])
    weed_img = np.rot90(weed_img, k)
    weed_mask = np.rot90(weed_mask, k)

    if random.random() > 0.5:
        weed_img = np.fliplr(weed_img)
        weed_mask = np.fliplr(weed_mask)

    if random.random() > 0.5:
        print("Applying random geometric transformation")
        weed_img, weed_mask = random_geometric_transform(weed_img, weed_mask)

    return weed_img, weed_mask


def inject_weeds(root="data/PhenoBench_aug/dreambooth_37"):
    root = Path(root)

    folder_crop_inpaintings = root / "inpainting_outputs"
    folder_crop_masks = root / "combined_masks"
    folder_weeds = root / "segmented_weeds"
    folder_weeds_masks = root / "segmented_weeds_masks"
    output_imgs = root / "final_images"
    output_masks = root / "final_masks"
    os.makedirs(output_masks, exist_ok=True)
    os.makedirs(output_imgs, exist_ok=True)

    # load weeds
    weeds = []
    for wf in os.listdir(folder_weeds):
        if not wf.endswith(".png"):
            continue
        path_img = os.path.join(folder_weeds, wf)
        path_mask = os.path.join(folder_weeds_masks, wf.replace(".png", "_mask.png"))

        weed_img = cv2.imread(path_img, cv2.IMREAD_UNCHANGED)
        weed_img = cv2.cvtColor(weed_img, cv2.COLOR_BGRA2RGBA)
        weed_mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)

        if weed_img is None or weed_mask is None or weed_img.shape[2] != 4:
            continue

        weeds.append((weed_img, weed_mask))

    image_files = [f for f in os.listdir(folder_crop_inpaintings) if f.endswith(".png")]
    random.shuffle(image_files)

    for filename in image_files:
        print(f"\nProcessing {filename}")
        image_path = os.path.join(folder_crop_inpaintings, filename)
        mask_path = os.path.join(
            folder_crop_masks, filename.replace(".png", "_mask.png")
        )

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        num_weeds = random.randint(0, 1)
        print(f"Adding {num_weeds} weeds...")

        if num_weeds == 1:
            weed_img, weed_mask = random.choice(weeds)
            weed_img, weed_mask = augment_weed(weed_img, weed_mask)
            cropped_img, cropped_mask = crop_to_content(weed_img, weed_mask)

            if cropped_img is None:
                continue

            position = find_background_spots(mask, cropped_mask)
            if position:
                image, mask = paste_weed(
                    image, mask, cropped_img, cropped_mask, position
                )
            else:
                print("No valid position found")

        Image.fromarray(image).save(os.path.join(output_imgs, filename))
        cv2.imwrite(
            os.path.join(output_masks, f"{filename.replace('.png', '_mask.png')}"), mask
        )


def weed_generation_pipeline(
    root="data/PhenoBench_aug/dreambooth_37",
    checkpoint="models/model_weeds",
    num_weeds=5,
):
    print("=== Generating weeds ===")
    # generate_weeds(checkpoint=checkpoint, root=root, num_images=num_weeds)

    print("\n=== Extracting weeds ===")
    extract_weeds(root=root)

    print("\n=== Injecting weeds into images ===")
    inject_weeds(root=root)
