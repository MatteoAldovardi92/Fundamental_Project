# Fundamental Project

## Overview
This repository contains the consolidated and optimized codebase for cross-task evaluation and training of the EoMT (Efficient Open-Vocabulary Multi-Task) model. It focuses on evaluating semantic and panoptic segmentation tasks across major datasets like Cityscapes and COCO.

## Directory Structure

- **`/eomt`**: The core model library. Contains the model architectures (`models/`), training loops (`training/`), and utilities.
- **`/eval`**: Custom evaluation scripts for metrics calculations (e.g., `iouEval.py`, `evalAnomaly.py`).
- **`/configs`**: YAML configuration files defining the model setup, data loaders, and hyperparameters for different datasets (e.g., DINOv2 on Cityscapes and COCO).
- **`/docs`**: Static files and HTML for the project's documentation and web presentation.
- **`/coco-classes-mapping-master`**: Tools and JSON mapping files used to translate COCO classes between the 80-class and 91-class formats.
- **`/gitfunctions`**: Utility scripts to help manage Git operations within Colab.

## Key Files
- **`Step4_fixed.ipynb` & `Step4_Professional_Cleaned.ipynb`**: Jupyter notebooks demonstrating the final optimized cross-task evaluation, utilizing the `eomt` module dynamically.
- **`requirements.txt`**: List of dependencies required to run the environment.

## Setup & Execution
1. Install dependencies: `pip install -r requirements.txt` (and ensure `lightning`, `jsonargparse` are installed).
2. Use the provided Jupyter Notebooks (e.g., `Step4_Professional_Cleaned.ipynb`) for inference, evaluation, and visualizations without manual model-rebuilding.

