# 🗄️ Dataset Loaders & Processing (`eomt/datasets/`)

*Disclaimer: This document is AI generated.*

This directory is absolutely critical for feeding data into your model properly. It handles the loading, parsing, and preprocessing of your raw datasets (like Cityscapes or COCO) before the tensors reach the neural network.

## 📝 Key Responsibilities:
- **PyTorch Dataset Classes (`Dataset`):** Custom classes that instruct PyTorch exactly how to read images and their corresponding ground-truth mask/label files from the disk directory.
- **Data Augmentation & Transforms:** Implementation of spatial transformations (like random cropping, resizing, horizontal flipping) and color jittering. This forces the model to learn robust features rather than memorizing exact pixels.
- **Label Mapping:** Converting raw pixel IDs from datasets into 'train IDs' used by the loss functions. For example, Cityscapes has 35 raw classes, but models are typically trained on a condensed set of 19 classes. This folder handles that ID translation.
- **Collation:** Preparing and padding batches of tensors (images, targets, and metadata) so they can be seamlessly processed by PyTorch `DataLoader` workers during the training loop.
