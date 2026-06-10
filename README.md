# Fundamental Project

## Overview
This repository contains the consolidated and optimized codebase for cross-task evaluation and training of the EoMT (Efficient Open-Vocabulary Multi-Task) model. It focuses on evaluating semantic and panoptic segmentation tasks across major datasets like Cityscapes and COCO.

## Directory Structure

- **`/eomt`**: The core model library. Contains the model architectures (`models/`), training loops (`training/`), dataset configs (`configs/`), and utilities.
- **`/eval`**: Custom evaluation scripts for metrics calculations (e.g., `iouEval.py`, `evalAnomaly.py`).
- **`/trained_models`**: Pre-trained and fine-tuned model weights (large files are git-ignored; back up to Google Drive).
- **`/best_eval_checkpoints`**: Best checkpoints selected after evaluation (git-ignored).
- **`/docs`**: Static files and HTML for the project's documentation and web presentation.
- **`/coco-classes-mapping-master`**: Tools and JSON mapping files used to translate COCO classes between the 80-class and 91-class formats.- **`/posthoc_metrics`**: Post-hoc metric computations and analysis scripts.
- **`/resultsAnomalyDetection`**: Saved results from anomaly detection evaluation runs.
- **`/saved_visualizations`**: Saved output visualizations from inference and evaluation.

## Key Files
- **`Step4.ipynb`**: Cross-task evaluation notebook — semantic and panoptic segmentation on Cityscapes and COCO.
- **`Step5.ipynb`**: LoRA fine-tuning notebook — fine-tunes EoMT on Cityscapes with three LoRA configurations.
- **`Step7-8.ipynb`**: Anomaly detection notebook — evaluates both ERFNet and fine-tuned EoMT on anomaly datasets.
- **`requirements.txt`**: List of dependencies required to run the environment.


