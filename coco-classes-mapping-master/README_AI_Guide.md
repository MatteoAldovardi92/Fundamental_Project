# 🗺️ COCO Classes Mapping (`coco-classes-mapping-master/`)

*Disclaimer: This document is AI generated.*

This folder contains specialized utilities for handling the often-confusing MS COCO dataset class indices, which is critical for evaluating object detection and instance/panoptic segmentation models.

## 📝 What this does & Why it matters:
The original MS COCO dataset annotation file contains 91 categories. However, most modern models are trained on a condensed subset of 80 categories (excluding classes that have very few annotations, like 'hat', 'shoe', or 'window').
- **Mapping Scripts:** Scripts like `map_coco_classes.py` translate model predictions back and forth between the 91-class format (needed for official COCO evaluation servers) and the 80-class format (used during actual PyTorch training).
- **JSON & Name Maps:** Contains `.json` mapping dictionaries (`coco_mapping_80to91.json`) and `.names` text files for easy class ID lookups during debugging, evaluation, or visualization.
- **Preventing Errors:** Without these files, a model predicting class ID '10' in the 80-class system might be incorrectly evaluated as a completely different object in the 91-class official ground truth, ruining your mAP (mean Average Precision) scores.
