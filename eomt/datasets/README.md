# eomt/datasets/

PyTorch Lightning DataModules and Dataset classes for all supported datasets.

## Files

| File | Dataset | Used in |
|---|---|---|
| `cityscapes_semantic.py` | Cityscapes (19 classes) | Step 4, Step 5 |
| `coco_panoptic.py` | COCO panoptic (133 classes) | upstream training |
| `coco_instance.py` | COCO instance | upstream training |
| `ade20k_semantic.py` / `ade20k_panoptic.py` | ADE20k | upstream training |
| `dataset.py` | Base dataset class | all of the above |
| `transforms.py` | Augmentations (crop, flip, blur, normalize) | all of the above |
| `lightning_data_module.py` | Base Lightning DataModule | all of the above |

## Cityscapes setup (Step 5)

Data is read directly from the original `.zip` files — no extraction needed.
Place the following in `eomt/data/` (git-ignored) or pass `path=` explicitly:

```
leftImg8bit_trainvaltest.zip
gtFine_trainvaltest.zip
```

On Colab the zips are copied to local SSD (`/content/cityscapes_data/`) at the
start of Step 5 to avoid slow Drive I/O during training.

## Augmentation (training)

- Random scale crop to 640×640
- Random horizontal flip
- Color jitter
- Gaussian blur (`p=0.5`) — enabled in all three LoRA experiments to address
  the sharpness distribution shift between COCO and Cityscapes
