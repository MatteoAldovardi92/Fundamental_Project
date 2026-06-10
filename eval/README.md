# eval/

Evaluation scripts for segmentation and anomaly detection.

## Files

| File | Description |
|---|---|
| `iouEval.py` | Per-class IoU accumulator used in Step 4 and Step 5 |
| `erfnet.py` / `erfnet_nobn.py` | ERFNet architecture (pixel-based baseline, Steps 7–8) |
| `evalAnomaly.py` | Anomaly segmentation evaluation script for ERFNet outputs |
| `eomt_anomaly_eval.py` | Anomaly evaluation adapted for EoMT mask-based outputs |
| `eval_iou.py` | IoU evaluation on Cityscapes val/train splits |
| `eval_cityscapes_color.py` | Saves colorized Cityscapes predictions for visualization |
| `eval_cityscapes_server.py` | Converts predictions to Cityscapes server submission format |
| `eval_forwardTime.py` | Measures model forward-pass time |
| `dataset.py` / `transform.py` | Dataset and transform helpers for ERFNet evaluation |

## Anomaly datasets (Steps 7–8)

Validation datasets are placed under `eval/Validation_Dataset/`:

```
eval/Validation_Dataset/
├── RoadAnomaly21/
├── RoadObstacle21/
├── FS_LostFound_full/
├── fs_static/
└── RoadAnomaly/
```

## Label convention (anomaly masks)

| Value | Meaning |
|---|---|
| 0 | In-distribution (normal) |
| 1 | Anomaly |
| 255 | Ignore |

`RoadAnomaly` (non-21) uses `2` for anomaly — remapped to `1` before metric computation.
