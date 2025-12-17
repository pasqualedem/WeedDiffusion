import os
import click

# =========================
# Root CLI
# =========================

@click.group()
def cli():
    """
    Main entry point.
    """
    pass


# =========================
# TRAIN COMMAND GROUP
# =========================

@cli.group()
def train():
    """
    Training commands.
    """
    pass


@train.command()
@click.argument("preset", type=click.Choice(["crop", "weed"]))
@click.argument("extra_args", nargs=-1)
def diffuser(preset, extra_args):
    """
    Train DreamBooth diffuser (crop / weed).
    """
    from weeddiff.train_diffuser import train as train_dreambooth

    train_dreambooth(
        preset=preset,
        extra_args=list(extra_args),
    )
    

@train.command()
@click.option("--config", type=click.Path(exists=True), required=True)
@click.option("--export_dir", type=click.Path(), required=True)
@click.option("--ckpt_path", type=click.Path(exists=True), required=False, default=None)
@click.option("--resume", is_flag=True, required=False, default=False)
def segmentor(config, export_dir, ckpt_path, resume):
    """
    Train Segmentor.
    """
    click.echo(f"Training Segmentor with config: {config}")
    
    from semantic_segmentation.train import main as train_segmentor
    train_segmentor(config_path=config, export_dir=export_dir, ckpt_path=ckpt_path, resume=resume)


# =========================
# GENERATE COMMAND
# =========================

@cli.command()
@click.argument("preset", type=click.Choice(["crop", "weed"]), required=True)
@click.option("--checkpoint", type=click.Path(exists=True), required=True)
@click.option("--root", type=click.Path(), required=True)
@click.option("--input_images", type=click.Path(), required=False, default="dreambooth/training/subset_crops")
@click.option("--num_weeds", required=False, default=5)
def generate(preset, checkpoint, root, input_images, num_weeds):
    """
    Generate outputs from a trained model.
    """
    if preset == "crop":
        from weeddiff.inpaint import inpaint_crops
        from weeddiff.mask_reconstruction import combine_images_to_masks

        inpaint_crops(root=root, checkpoint=checkpoint, input_images=input_images)
        combine_images_to_masks(root=root)
        
    elif preset == "weed":
        from weeddiff.generate_weed import weed_generation_pipeline

        weed_generation_pipeline(root=root, checkpoint=checkpoint, num_weeds=num_weeds)
    else:
        raise ValueError(f"Unknown preset: {preset}")


# =========================
# TEST COMMAND (SEGMENTOR ONLY)
# =========================

@cli.command()
@click.option("--config", type=click.Path(exists=True), required=True)
@click.option("--ckpt_path", type=click.Path(exists=True), required=True)
@click.option("--export_dir", type=click.Path(exists=True), required=True)
def test(config, ckpt_path, export_dir):
    """
    Test Segmentor only.
    """
    from semantic_segmentation.test import test_segmentor
    
    click.echo(
        f"Testing Segmentor with checkpoint={ckpt_path}, export_dir={export_dir}"
    )
    test_segmentor(config_path=config, ckpt_path=ckpt_path, export_dir=export_dir)


# =========================
# EXPERIMENT COMMAND (SEGMENTOR ONLY)
# =========================

@cli.command()
@click.option("--config", type=click.Path(exists=True), required=True)
@click.option("--export_dir", type=click.Path(), required=True)
@click.option("--ckpt_path", type=click.Path(exists=True), required=False, default=None)
@click.option("--resume", is_flag=True, required=False, default=False)
def experiment(config, export_dir, ckpt_path, resume):
    """
    Run full experiment: train + test Segmentor.
    """
    from semantic_segmentation.train import train_segmentor
    from semantic_segmentation.test import test_segmentor
    
    click.echo(f"Running full experiment with config: {config}")

    train_export_dir = os.path.join(export_dir, "train")
    test_export_dir = os.path.join(export_dir, "test")
    
    train_segmentor(config_path=config, export_dir=train_export_dir, ckpt_path=ckpt_path, resume=resume)
    
    # Get the best checkpoint from training
    ckpt_paths = os.path.join(
        train_export_dir,
        "lightning_logs",
    )
    # Get last version
    versions = [
        d for d in os.listdir(ckpt_paths) if d.startswith("version_")
    ]
    versions.sort(key=lambda x: int(x.split("_")[-1]))
    last_version = versions[-1]
    print(f"Using checkpoint from version: {last_version}")
    checkpoints_dir = os.path.join(
        ckpt_paths,
        last_version,
        "checkpoints",
    )
    # Get best checkpoint (it has mIoU in the filename)
    best_ckpt = [f for f in os.listdir(checkpoints_dir) if "train_mIoU" in f]
    assert len(best_ckpt) > 0, "No checkpoint found!"
    assert len(best_ckpt) == 1, "Multiple checkpoints found!"
    best_ckpt_path = os.path.join(checkpoints_dir, best_ckpt[0])
    print(f"Best checkpoint path: {best_ckpt_path}")
    
    test_segmentor(config_path=config, ckpt_path=best_ckpt_path, export_dir=test_export_dir)


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    cli()
