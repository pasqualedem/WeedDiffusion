from weeddiff.dreambooth import parse_args, main as dreambooth_main


BASE_ARGS = [
    "--pretrained_model_name_or_path", "runwayml/stable-diffusion-v1-5",
    "--resolution", "512",
    "--train_batch_size", "1",
    "--gradient_accumulation_steps", "1",
    "--learning_rate", "1e-6",
    "--lr_scheduler", "constant",
    "--lr_warmup_steps", "0",
    "--max_train_steps", "4000",
    "--train_text_encoder",
    "--mixed_precision", "fp16",
    "--gradient_checkpointing",
    "--checkpointing_steps", "1000",
]


PRESETS = {
    "crop": {
        "instance_dir": "dreambooth/training/subset_crops",
        "output_dir": "out/model_crops",
        "instance_prompt": "sks crop",
    },
    "weed": {
        "instance_dir": "dreambooth/training/subset_weeds",
        "output_dir": "out/model_weeds",
        "instance_prompt": "sks weed",
    },
}


def train(
    *,
    preset: str,
    extra_args: list[str] | None = None,
):
    """
    Generic DreamBooth trainer.
    """
    if preset not in PRESETS:
        raise ValueError(
            f"Invalid preset '{preset}'. "
            f"Allowed: {list(PRESETS.keys())}"
        )

    if extra_args is None:
        extra_args = []

    p = PRESETS[preset]

    preset_args = BASE_ARGS + [
        "--instance_data_dir", p["instance_dir"],
        "--output_dir", p["output_dir"],
        "--instance_prompt", p["instance_prompt"],
    ]

    # Presets first, user args override
    all_args = preset_args + extra_args

    parsed_args = parse_args(all_args)
    dreambooth_main(parsed_args)
