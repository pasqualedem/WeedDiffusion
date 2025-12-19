python main.py experiment --config ./semantic_segmentation/config/images_37/config_erfnet_geocolor.yaml --export_dir out/segmentation/images_37/erfnet_geocolor
python main.py experiment --config ./semantic_segmentation/config/images_37/config_erfnet_base.yaml     --export_dir out/segmentation/images_37/erfnet_base
python main.py experiment --config ./semantic_segmentation/config/images_37/config_erfnet_geo.yaml      --export_dir out/segmentation/images_37/erfnet_geo
python main.py experiment --config ./semantic_segmentation/config/images_37/config_erfnet_color.yaml    --export_dir out/segmentation/images_37/erfnet_color

python main.py experiment --config ./semantic_segmentation/config/images_37/config_deeplab_geocolor.yaml --export_dir out/segmentation/images_37/deeplab_geocolor
python main.py experiment --config ./semantic_segmentation/config/images_37/config_deeplab_base.yaml     --export_dir out/segmentation/images_37/deeplab_base
python main.py experiment --config ./semantic_segmentation/config/images_37/config_deeplab_geo.yaml      --export_dir out/segmentation/images_37/deeplab_geo
python main.py experiment --config ./semantic_segmentation/config/images_37/config_deeplab_color.yaml    --export_dir out/segmentation/images_37/deeplab_color

python main.py experiment --config ./semantic_segmentation/config/images_37/config_unet_geocolor.yaml --export_dir out/segmentation/images_37/unet_geocolor
python main.py experiment --config ./semantic_segmentation/config/images_37/config_unet_base.yaml     --export_dir out/segmentation/images_37/unet_base #
python main.py experiment --config ./semantic_segmentation/config/images_37/config_unet_geo.yaml      --export_dir out/segmentation/images_37/unet_geo
python main.py experiment --config ./semantic_segmentation/config/images_37/config_unet_color.yaml    --export_dir out/segmentation/images_37/unet_color #


# WeedDiffusion

python main.py experiment --config semantic_segmentation/config/weeddiff_37/config_erfnet_geocolor.yaml --export_dir out/segmentation/weeddiff_37/erfnet_geocolor
python main.py experiment --config semantic_segmentation/config/weeddiff_37/config_erfnet_base.yaml     --export_dir out/segmentation/weeddiff_37/erfnet_base