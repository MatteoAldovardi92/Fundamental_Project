# 🏋️ Training Logic & Lightning Modules (`eomt/training/`)

*Disclaimer: This document is AI generated.*

This folder abstracts the complex PyTorch boilerplate loops (like moving tensors to GPUs, zeroing gradients, and step tracking) using PyTorch Lightning.

## 📝 Key Scripts:
- **`lightning_module.py`**: The core PyTorch Lightning module. It defines what happens during `training_step`, `validation_step`, and configures the optimizers and learning rate schedulers.
- **`mask_classification_semantic.py`**: Defines the specific logic, loss calculations, and metrics needed for Semantic Segmentation (classifying every pixel into categories like 'road', 'car', 'tree').
- **`mask_classification_panoptic.py`**: Handles Panoptic Segmentation logic, which is a step beyond semantic—it not only identifies pixels but separates distinct instances of objects (e.g., 'car 1' vs 'car 2').
- **`mask_classification_instance.py`**: Handles pure Instance Segmentation tasks.
- **`mask_classification_loss.py`**: Centralizes the mathematical loss functions used to train the segmentation models.
- **`two_stage_warmup_poly_schedule.py`**: A custom learning rate scheduler often used in computer vision to slowly warm up the learning rate before decaying it.
