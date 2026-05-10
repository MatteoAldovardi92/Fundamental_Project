# ⚙️ Training Configurations (`eomt/configs/`)

*Disclaimer: This document is AI generated.*

Instead of hardcoding batch sizes, learning rates, and dataset paths directly into Python scripts, machine learning projects use configuration files. This makes running multiple different experiments much easier.

## 📝 How it works:
- You will typically find `.yaml` files in subdirectories here (e.g., `dinov2/cityscapes/semantic/eomt_base_640.yaml`).
- When you run a training script (like `main.py`), you will pass one of these config files as an argument.
- These configs define:
  - Which dataset to use (e.g., Cityscapes).
  - The image resolution (e.g., 640x640).
  - Training hyperparameters (epochs, learning rate, weight decay).
  - Model specific settings (ViT base vs ViT large).
