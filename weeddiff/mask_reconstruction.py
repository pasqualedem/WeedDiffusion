import os
from pathlib import Path
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


color_map = {
    0: (0, 0, 139),      # background -> dark blue
    1: (0, 255, 0),      # crop -> green
    2: (255, 0, 0),      # weed -> red
    3: (0, 128, 0),      # partial crop -> dark green
    4: (128, 0, 0)       # partial weed -> dark red
}

def apply_color_map(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        color_mask[mask == class_id] = color
    return color_mask


def get_aug_info(filename):
    base = filename.lower().replace("_upscayl", "")
    if "left" in base:
        direction = "left"
    elif "right" in base:
        direction = "right"
    else:
        return None, None, None

    try:
        perc = int(base.split(f"{direction}_")[1].split(".")[0])
    except:
        return None, None, None

    base_name = base.split(f"_{direction}_")[0] 
    base_name = base_name.upper()
    base_name += ".png"
    return base_name, direction, perc


def extract_green_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def combine_masks(original_mask, green_mask, direction, perc):
    h, w = original_mask.shape
    slice_width = int(w * (perc / 100))
    updated_mask = original_mask.copy()

    # In the augmented area, green pixels are set to 1 (crop), everything else is set to 0 (background)
    # The non-augmented area remains original
    if direction == "left":
        area = slice(0, slice_width)
        updated_mask[:, area] = np.where(green_mask[:, area] > 0, 1, 0)

    elif direction == "right":
        area = slice(w - slice_width, w)
        updated_mask[:, area] = np.where(green_mask[:, area] > 0, 1, 0)

    return updated_mask


def combine_images_to_masks(root="data/PhenoBench_aug/dreambooth_37"):

    folder_images_orig = Path("data/PhenoBench/train/images")
    folder_aug = Path(root) / "inpainting_outputs"
    folder_output = Path(root) / "combined_masks"
    
    
    folder_masks_orig = Path("data/PhenoBench/train/semantics")
    folder_masks_comb = Path(root) / "combined_masks"
    
    os.makedirs(folder_output, exist_ok=True)
    
    # loop through augmented images
    for filename in tqdm(os.listdir(folder_aug)):
        if not filename.lower().endswith(".png"):
            continue

        filename_original = filename  
        aug_path = os.path.join(folder_aug, filename_original)
        base_name, direction, perc = get_aug_info(filename_original)

        if not base_name or not direction:
            print(f"❌ Skipping malformed filename: {filename}")
            continue

        original_img_path = os.path.join(folder_images_orig, base_name)
        original_mask_path = os.path.join(folder_masks_orig, base_name)

        if not os.path.exists(original_img_path) or not os.path.exists(original_mask_path):
            print(f"❌ Missing original image or mask for: {base_name}")
            continue

        # Load images
        aug_img = cv2.imread(aug_path)
        aug_img_rgb = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
        original_mask = cv2.imread(original_mask_path, cv2.IMREAD_UNCHANGED)

        # Calculate green mask on augmented image
        green_mask = extract_green_mask(aug_img_rgb)

        # Combine masks (replace crop in modified area)
        combined_mask = combine_masks(original_mask, green_mask, direction, perc)

        # Save
        output_path = os.path.join(folder_masks_comb, filename_original.replace(".png", "_mask.png"))
        cv2.imwrite(output_path, combined_mask)

    # Take 5 random .png files from "total photos"
    all_aug_files = [f for f in os.listdir(folder_aug) if f.endswith(".png")]
    sampled_files = random.sample(all_aug_files, 3)

    for filename in sampled_files:
        aug_path = os.path.join(folder_aug, filename)

        # Extract base name (remove everything after "_left_X" or "_right_X")
        base_name = filename
        if "_left_" in base_name:
            base_name = base_name.split("_left_")[0] 
            base_name = base_name.upper() + ".png"
        elif "_right_" in base_name:
            base_name = base_name.split("_right_")[0] 
            base_name = base_name.upper() + ".png"

        orig_img_path = os.path.join(folder_images_orig, base_name)
        orig_mask_path = os.path.join(folder_masks_orig, base_name)
        combined_mask_path = os.path.join(folder_masks_comb, filename.replace(".png", "_mask.png"))

        if not os.path.exists(orig_img_path) or not os.path.exists(orig_mask_path) or not os.path.exists(combined_mask_path):
            print(f"❌ Dati mancanti per: {filename}")
            continue

        # Load images and masks
        img_aug = cv2.imread(aug_path)

        img_orig = cv2.imread(orig_img_path)
        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

        mask_orig = cv2.imread(orig_mask_path, cv2.IMREAD_UNCHANGED)
        mask_comb = cv2.imread(combined_mask_path, cv2.IMREAD_UNCHANGED)

        # Apply color map
        color_mask_orig = apply_color_map(mask_orig)
        color_mask_comb = apply_color_map(mask_comb)

        # Plotting
        plt.figure(figsize=(7, 8))
        plt.suptitle(f"File: {filename}", fontsize=14)

        plt.subplot(2, 2, 1)
        plt.imshow(img_orig)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(2, 2, 2)
        plt.imshow(img_aug)
        plt.title("Augmented Image")
        plt.axis("off")

        plt.subplot(2, 2, 3)
        plt.imshow(color_mask_orig)
        plt.title("Original Mask")
        plt.axis("off")

        plt.subplot(2, 2, 4)
        plt.imshow(color_mask_comb)
        plt.title("Combined Mask")
        plt.axis("off")

        plt.tight_layout()
        plt.show()