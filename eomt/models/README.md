# eomt/models/

Neural network architecture definitions.

## Files

| File | Description |
|---|---|
| `vit.py` | ViT backbone wrapping `timm`'s `vit_base_patch14_reg4_dinov2` |
| `eomt.py` | Full EoMT model — ViT encoder + mask/class decoder heads |
| `scale_block.py` | Decoder scale blocks for multi-scale feature fusion |

## EoMT architecture

```
Input image (640×640)
    └── ViT encoder (12 blocks, patch size 14, DINOv2 weights)
            └── num_blocks decoder scale blocks
                    ├── mask queries (num_q × H/4 × W/4)
                    └── class queries (num_q × num_classes+1)
```
