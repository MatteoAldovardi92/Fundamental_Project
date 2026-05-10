# 🚀 Fundamental ML Project - Beginner's Guide

*Disclaimer: This document is AI generated.*

Welcome to the Fundamental Project! This repository is focused on Machine Learning model evaluation and fine-tuning (e.g., semantic segmentation, LoRA experiments, and anomaly detection).

## ⚠️ Important Note for Novices
**This repository is NOT self-contained.** To keep the repository lightweight and comply with Git limits, **large files have been excluded**.
- **Missing Datasets:** Datasets (like the 11GB Cityscapes data) are ignored. You must retrieve them locally into the `eomt/data/` folder using the provided download notebook cells.
- **Missing Model Weights:** Pre-trained model weights (`.pth`, `.safetensors`, etc.) are also excluded. Ensure you place your weights in the appropriate local directories before running inference.

## 📂 Repository Structure
- **`eval/`**: Contains scripts to evaluate model performance (IoU metrics, anomaly detection).
- **`eomt/`**: Contains the core model architectures, configurations, and training pipelines.
- **`gitfunctions/`**: Helper scripts designed to make Git commands easier from Google Colab.
- **`coco-classes-mapping-master/`**: Utilities for handling COCO dataset class indices.
- **`trained_models/`**: Designated storage for model weights.

Dive into each subfolder to find its specific `README_AI_Guide.md` for more details on the individual scripts!
