# coco-classes-mapping-master/

Tools and mappings to convert COCO panoptic predictions into Cityscapes classes for zero-shot cross-dataset evaluation.

## Files

| File | Description |
|---|---|
| `bridge_to_cs.py` | Generates the COCO→Cityscapes class mapping by stripping suffixes (e.g. `wall-brick` → `wall`) and applying a synonym dictionary (e.g. `tree` → `vegetation`) |
| `panoptic_coco_categories.json` | Official list of 133 COCO panoptic categories, used as the source of truth for mapping model output indices |
| `coco_to_cs.json` | Final numerical mapping produced by `bridge_to_cs.py`: dense COCO indices (0–132) → Cityscapes class IDs (0–18), used as an O(1) GPU lookup table during evaluation |
