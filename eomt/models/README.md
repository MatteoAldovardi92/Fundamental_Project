# eomt/models/

Neural network architecture definitions.

## Files

| File | Description |
|---|---|
| `vit.py` | ViT backbone wrapping `timm`'s `vit_base_patch14_reg4_dinov2` |
| `eomt.py` | Full EoMT model — ViT encoder + mask/class decoder heads |
| `scale_block.py` | Decoder scale blocks used by EoMT for multi-scale feature fusion |

## EoMT architecture

```
Input image (640×640)
    └── ViT encoder (12 blocks, patch size 14, DINOv2 weights)
            └── num_blocks decoder scale blocks
                    ├── mask queries (num_q × H/4 × W/4)
                    └── class queries (num_q × num_classes+1)
```

**Key instantiation parameters:**

| Parameter | Step 5 value | Note |
|---|---|---|
| `num_classes` | 19 | Cityscapes semantic |
| `num_q` | 200 | Fixed by COCO checkpoint shape |
| `num_blocks` | 3 | Decoder depth |
| `masked_attn_enabled` | False | Disabled for LoRA fine-tuning |

`num_q=200` cannot be changed when loading the COCO checkpoint — doing so
causes a shape mismatch on `network.q.weight` even with `strict=False`.
