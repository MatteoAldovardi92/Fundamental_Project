# 🏗️ Model Architectures (`eomt/models/`)

*Disclaimer: This document is AI generated.*

This directory defines the actual structure of the neural networks you are training or fine-tuning.

## 📝 Key Scripts:
- **`vit.py`**: Implements the Vision Transformer (ViT) backbone. This is often the foundational model that extracts features from your images.
- **`scale_block.py`**: Defines scaling modules or intermediate blocks (potentially used for adapter/LoRA integration) that adjust the features passing through the network.
- **`eomt.py`**: The overarching wrapper script that ties the backbone (like ViT) together with the task-specific heads (like a segmentation head) to build the complete, end-to-end model.
