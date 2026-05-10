# 📊 Evaluation Module (`eval/`)

*Disclaimer: This document is AI generated.*

This module contains all the necessary scripts to assess the performance of the trained computer vision models.

## 📝 Key Scripts:
- **`dataset.py` & `transform.py`**: Handles loading and preprocessing/augmenting the evaluation datasets (like Cityscapes).
- **`erfnet.py` & `erfnet_nobn.py`**: Defines the ERFNet architecture (Efficient Residual Factorized ConvNet) used for real-time semantic segmentation during evaluation.
- **`evalAnomaly.py`**: Script dedicated to detecting out-of-distribution or anomalous objects in the scene.
- **`eval_iou.py` & `iouEval.py`**: Calculates the Intersection over Union (IoU) metric, which is the standard measure for semantic segmentation accuracy.
- **`eval_cityscapes_color.py` & `eval_cityscapes_server.py`**: Specific evaluation pipelines tailored to the Cityscapes dataset formats.
