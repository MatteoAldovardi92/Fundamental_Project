# trained_models/

Pre-trained and fine-tuned model weights.

## Contents

| File | Description |
|---|---|
| `erfnet_pretrained.pth` | ERFNet encoder — pixel-based baseline for Steps 7–8 |

Large files (`.pth`, `.ckpt`, `.bin`) are git-ignored. Back up to Google Drive.

## Fine-tuned checkpoints (Step 5)

Saved to `checkpoints/<experiment_name>/` at the project root (also git-ignored):

```
checkpoints/
├── lora-head-only/
│   ├── last.ckpt
│   └── eomt-epoch=21-losses/val_loss_total=2.239.ckpt  ← top-1
├── lora-decoder-blocks/
│   ├── last.ckpt
│   └── eomt-epoch=21-losses/val_loss_total=2.138.ckpt  ← top-1
└── lora-all-blocks/
    ├── last.ckpt
    └── eomt-epoch=23-losses/val_loss_total=2.009.ckpt  ← top-1
```

The best overall checkpoint (`lora-all-blocks` top-1, mIoU 75.39%) is also
saved as `best_eval_checkpoints/OVERALL_CHAMPION_Exp3_mIoU_75.39.ckpt` and
used in Steps 7–8 as the `Fine-tuned` model.
