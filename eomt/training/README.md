# eomt/training/

PyTorch Lightning modules, loss, LR scheduler, and LoRA fine-tuning logic.

## Files

| File | Description |
|---|---|
| `lightning_module.py` | Base Lightning module — optimizer, scheduler, metrics, visualization |
| `mask_classification_semantic.py` | Semantic segmentation — `eval_step`, mIoU metrics, windowed inference |
| `mask_classification_panoptic.py` | Panoptic segmentation — PQ/SQ/RQ metrics |
| `mask_classification_instance.py` | Instance segmentation — mAP metrics |
| `mask_classification_lora.py` | **LoRA fine-tuning** — used in Step 5 |
| `mask_classification_loss.py` | Hungarian matching loss (Mask2Former-style) |
| `two_stage_warmup_poly_schedule.py` | Two-stage warmup + polynomial LR decay scheduler |

## MaskClassificationLoRA (Step 5)

Extends `MaskClassificationSemantic`. Key differences from the base class:

- Injects PEFT LoRA adapters into the ViT encoder after loading the COCO checkpoint
- `init_metrics_semantic` → empty `ModuleList` (no mIoU during training, avoids extra progress bar)
- `validation_step` → logs `val_loss_total`, `val_loss_mask`, `val_loss_dice`, `val_cross_entropy`
- `configure_optimizers` → LLRD + weight decay exclusion for biases/norms

## LR schedule

`TwoStageWarmupPolySchedule` applies per param group:

```
Non-backbone params:  [0 → base_lr] over warmup_steps[0], then poly decay
Backbone params:      frozen during warmup_steps[0], [0 → base_lr] over
                      warmup_steps[1], then poly decay
```

With `warmup_steps=[200, 200]` and ~3725 total optimizer steps (25 epochs):
- Non-backbone warmup ends at step 200 (5.4%)
- Backbone warmup ends at step 400 (10.7%)

## Loss

`MaskClassificationLoss` wraps `Mask2Former`'s Hungarian matcher + loss.
Called 3× per training batch (once per decoder block). Components:

| Key | Coefficient |
|---|---|
| `loss_mask` (BCE) | 5.0 |
| `loss_dice` | 5.0 |
| `cross_entropy` | 2.0 |
