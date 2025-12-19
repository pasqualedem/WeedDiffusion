import os
import shutil

subset_37 = [
    "05-15_00061_P0030852.png",
    "05-15_00082_P0030852.png",
    "05-15_00095_P0030852.png",
    "05-15_00096_P0030855.png",
    "05-15_00097_P0030855.png",
    "05-15_00098_P0030852.png",
    "05-15_00116_P0030947.png",
    "05-15_00117_P0030852.png",
    "05-15_00119_P0030692.png",
    "05-15_00124_P0030852.png",
    "05-15_00132_P0030690.png",
    "05-15_00133_P0030855.png",
    "05-15_00137_P0030947.png",
    "05-15_00141_P0030690.png",
    "05-15_00142_P0030690.png",
    "05-15_00156_P0030947.png",
    "05-15_00179_P0030949.png",
    "05-26_00102_P0034280.png",
    "05-26_00195_P0034117.png",
    "05-26_00214_P0034117.png",
    "05-26_00215_P0034117.png",
    "05-26_00242_P0034119.png",
    "05-26_00250_P0034020.png",
    "05-26_00264_P0034119.png",
    "05-26_00270_P0034020.png",
    "05-26_00271_P0034020.png",
    "06-05_00051_P0037987.png",
    "06-05_00061_P0037989.png",
    "06-05_00072_P0037987.png",
    "06-05_00093_P0037987.png",
    "06-05_00120_P0037989.png",
    "06-05_00161_P0038055.png",
    "06-05_00178_P0038055.png",
    "06-05_00180_P0037818.png",
    "06-05_00182_P0038055.png",
    "06-05_00203_P0038055.png",
    "06-05_00224_P0038055.png",
]


def make_subset():
    src_dir = "data/PhenoBench/train/images"
    dst_dir = "data/PhenoBench_subset/train/images_37"

    os.makedirs(dst_dir, exist_ok=True)

    for fname in subset_37:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)

        if not os.path.isfile(src):
            raise FileNotFoundError(f"Missing file: {src}")

        shutil.copy2(src, dst)

    print(f"Copied {len(subset_37)} files successfully.")
