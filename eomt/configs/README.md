# eomt/configs/

YAML configuration files for training and fine-tuning runs.

## Structure

```
configs/
├── dinov2/
│   ├── cityscapes/semantic/eomt_base_640.yaml   — upstream Cityscapes training
│   └── coco/panoptic/eomt_base_640_2x.yaml      — upstream COCO training
└── experiments/
    ├── exp1_lora_head_only.yaml       — Step 5 Exp 1: decoder heads only
    ├── exp2_lora_decoder_blocks.yaml  — Step 5 Exp 2: heads + LoRA blocks 9–11
    ├── exp3_lora_all_blocks.yaml      — Step 5 Exp 3: heads + LoRA all 12 blocks
    ├── optimal_lora_config.yaml       — best config from ablation search
    └── EXPERIMENT_RATIONALE.md        — ablation rationale with references
```
