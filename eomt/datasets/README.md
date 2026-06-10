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
