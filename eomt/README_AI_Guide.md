# 🧠 End-of-Module Training (`eomt/`)

*Disclaimer: This document is AI generated.*

This folder holds the core training logic, model architectures, and experimental setups (such as LoRA fine-tuning workflows) for the project.

## 📝 Folder Breakdown:
- **`models/`**: Defines the neural network architecture (e.g., Vision Transformers). Check `eomt/models/README_AI_Guide.md` for details.
- **`training/`**: Contains the PyTorch Lightning trainer logic and loss functions. Check `eomt/training/README_AI_Guide.md` for details.
- **`configs/`**: Stores `.yaml` configuration files that govern hyperparameters. Check `eomt/configs/README_AI_Guide.md` for details.
- **`datasets/`**: Handles the data loading pipelines for PyTorch. Check `eomt/datasets/README_AI_Guide.md` for details.
- **`docs/`**: Web assets for project showcase. Check `eomt/docs/README_AI_Guide.md` for details.
- **`data/`**: The designated folder for heavy datasets (e.g., Cityscapes). *Note: Contents of this folder are ignored by git to save space.*
- **`main.py`**: The primary entry point for kicking off a training or fine-tuning run.
