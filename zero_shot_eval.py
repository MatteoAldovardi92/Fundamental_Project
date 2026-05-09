import torch
import yaml
import json
import os
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from models.eomt import EoMT
import importlib

# --- 1. Load the Mapping Dictionaries ---
mapping_file = "coco_mapping_80to91.json"
with open(mapping_file, "r") as f:
    coco_map = {int(k): int(v) for k, v in json.load(f).items()}

bridge = {1: 11, 2: 18, 3: 13, 4: 17, 6: 15, 7: 16, 8: 14, 10: 6, 13: 7}
stuff_bridge = {100: 0, 123: 1, 91: 2, 129: 2, 109: 3, 110: 3, 111: 3, 112: 3, 131: 3, 117: 4, 116: 8, 125: 8, 126: 9, 119: 10}

def map_to_cityscapes(pred_tensor):
    mapped = torch.full_like(pred_tensor, 19)
    for model_id, coco91_id in coco_map.items():
        if coco91_id in bridge:
            mapped[pred_tensor == (model_id - 1)] = bridge[coco91_id]
    for stuff_id, cs_id in stuff_bridge.items():
        mapped[pred_tensor == stuff_id] = cs_id
    return mapped

# --- 2. Load Model and DataModule ---
def load_components():
    # Model
    with open("configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml", "r") as f:
        cfg_m = yaml.safe_load(f)
    
    from models.vit import ViT
    encoder = ViT(img_size=(640, 640), **cfg_m["model"]["init_args"]["network"]["init_args"]["encoder"]["init_args"])
    network = EoMT(encoder=encoder, num_classes=133, **{k:v for k,v in cfg_m["model"]["init_args"]["network"]["init_args"].items() if k != "encoder"})
    
    model_path, model_class = cfg_m["model"]["class_path"].rsplit(".", 1)
    kwargs = {k:v for k,v in cfg_m["model"]["init_args"].items() if k != "network"}
    model = getattr(importlib.import_module(model_path), model_class)(
        network=network, img_size=(640,640), num_classes=133, stuff_classes=list(range(80, 133)), **kwargs
    )
    state_dict = torch.load("eomt_coco.bin", map_location="cpu")
    if "state_dict" in state_dict: state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)

    # DataModule
    from datasets.cityscapes_semantic import CityscapesSemantic
    dm = CityscapesSemantic(path="./data", img_size=(640, 640), batch_size=1, check_empty_targets=False)
    return model, dm

model, dm = load_components()

# --- 3. Monkey-Patch the Validation Step ---
from eval.iouEval import iouEval
evaluator = iouEval(20)

def custom_validation_step(self, batch, batch_idx):
    import torch.nn.functional as F
    imgs, targets = batch
    transformed = self.resize_and_pad_imgs_instance_panoptic([imgs[0]])
    m_logits, c_logits = self(transformed)
    m_logits = F.interpolate(m_logits[-1], self.img_size, mode="bilinear")
    m_logits = self.revert_resize_and_pad_logits_instance_panoptic(m_logits, [imgs[0].shape[-2:]])
    
    pred = self.to_per_pixel_preds_panoptic(m_logits, c_logits[-1], self.stuff_classes, 0.8, 0.8)[0][..., 0]
    mapped_pred = map_to_cityscapes(pred)
    
    gt = torch.full(imgs[0].shape[-2:], 19, device=mapped_pred.device)
    for m, l in zip(targets[0]['masks'], targets[0]['labels']):
        gt[m] = l
    gt[gt == 255] = 19
    
    evaluator.addBatch(mapped_pred.unsqueeze(0).unsqueeze(0), gt.unsqueeze(0).unsqueeze(0))

# Apply the patch
import types
model.validation_step = types.MethodType(custom_validation_step, model)

# --- 4. Run Evaluation ---
trainer = Trainer(accelerator="gpu", devices=1, logger=False)
trainer.validate(model, datamodule=dm)

_, iou_classes = evaluator.getIoU()
miou = iou_classes[:19].mean()
print(f"\n>>> CUSTOM EVAL SCRIPT: Final Mapped mIoU = {miou*100:.2f}%")
