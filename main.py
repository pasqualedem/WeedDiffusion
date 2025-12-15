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
def segmentor(config):
    """
    Train Segmentor.
    """
    click.echo(f"Training Segmentor with config: {config}")
    # train_segmentor(config)


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
@click.option("--checkpoint", type=click.Path(exists=True), required=True)
@click.option("--dataset", type=click.Path(exists=True), required=True)
def test(checkpoint, dataset):
    """
    Test Segmentor only.
    """
    click.echo(
        f"Testing Segmentor with checkpoint={checkpoint}, dataset={dataset}"
    )
    # test_segmentor(checkpoint, dataset)


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    cli()
