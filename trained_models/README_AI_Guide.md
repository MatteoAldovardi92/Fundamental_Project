# 💾 Pre-trained Models (`trained_models/`)

*Disclaimer: This document is AI generated.*

This directory is designated for storing serialized neural network weights (the actual "learned" knowledge of the models).

## 📝 Important Details:
- **Large Files Ignored:** To prevent Git from crashing, large model files (like `.pth`, `.safetensors`, `.bin`) are generally ignored by `.gitignore`. 
- **Existing Files:** You might see smaller or essential base files here, such as `erfnet_encoder_pretrained.pth.tar`, which acts as a starting point (encoder backbone) before fine-tuning.
- **Workflow:** When you train a model or apply LoRA in this Colab environment, save your final `.pth` files here. However, remember to back them up to your personal Google Drive, as pushing massive files to GitHub will fail.
