# eomt/configs/

YAML configuration files for training and fine-tuning runs.

## Structure

```
configs/
├── dinov2/
│   ├── cityscapes/semantic/eomt_base_640.yaml   — upstream Cityscapes training
│   └── coco/panoptic/eomt_base_640_2x.yaml      — upstream COCO training
└── experiments/
    ├── exp1_lora_head_only.yaml       — Step 5 Exp 1: head only
    ├── exp2_lora_decoder_blocks.yaml  — Step 5 Exp 2: heads + LoRA blocks 9–11
    ├── exp3_lora_all_blocks.yaml      — Step 5 Exp 3: heads + LoRA all 12 blocks
    ├── optimal_lora_config.yaml       — best config from ablation search
    └── EXPERIMENT_RATIONALE.md        — full ablation rationale with references
```

## Shared fine-tuning hyperparameters

| Parameter | Value |
|---|---|
| `lr` | 1e-4 |
| `llrd` | 0.9 |
| `weight_decay` | 0.05 |
| `warmup_steps` | [200, 200] |
| `poly_power` | 0.9 |
| `lora_r` | 8 |
| `lora_alpha` | 16 |
| `lora_target_modules` | qkv, proj, fc1, fc2 |
| `batch_size` | 4 (effective 16 with grad accum) |
| `max_epochs` | 25 |
