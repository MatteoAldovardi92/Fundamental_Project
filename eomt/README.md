# EoMT

Adapted version of the [original EoMT repository](https://github.com/tue-mps/eomt). Contains the full model library for training and evaluating the EoMT (Efficient Open-Vocabulary Multi-Task) model on semantic and panoptic segmentation tasks.

A pre-trained EoMT model on Cityscapes is available on [Google Drive](https://drive.google.com/drive/folders/1q2vHUzora2nP52fP50zmoQAykWuwoGav?usp=drive_link).

## Directory Structure

- **`models/`**: ViT backbone and full EoMT architecture definitions.
- **`training/`**: PyTorch Lightning modules, loss functions, LR scheduler, and LoRA fine-tuning logic.
- **`datasets/`**: DataModules and Dataset classes for Cityscapes, COCO, and ADE20k.
- **`configs/`**: YAML configuration files for upstream training and LoRA fine-tuning experiments.
- **`data/`**: Dataset zip files (git-ignored).
- **`main.py`**: Entry point for training and evaluation via the Lightning CLI.
- **`requirements.txt`**: Python dependencies for this module.
- **`inference.ipynb`**: Notebook for quick inference and visualization with auto-downloaded pre-trained models.
