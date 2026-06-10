# eomt/training/

PyTorch Lightning modules, loss function, LR scheduler, and LoRA fine-tuning logic.

## Files

| File | Description |
|---|---|
| `lightning_module.py` | Base Lightning module — optimizer, scheduler, metrics, visualization |
| `mask_classification_semantic.py` | Semantic segmentation module with mIoU metrics and windowed inference |
| `mask_classification_panoptic.py` | Panoptic segmentation module with PQ/SQ/RQ metrics |
| `mask_classification_instance.py` | Instance segmentation module with mAP metrics |
| `mask_classification_lora.py` | LoRA fine-tuning module used in Step 5 — extends `MaskClassificationSemantic` |
| `mask_classification_loss.py` | Hungarian matching loss (Mask2Former-style) with BCE, Dice, and cross-entropy components |
| `two_stage_warmup_poly_schedule.py` | Two-stage warmup + polynomial LR decay scheduler |
