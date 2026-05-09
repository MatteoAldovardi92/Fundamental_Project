import os
import sys
import json
import torch
import torch.nn.functional as F

# Suppress PyTorch FX warnings for DINOv3 models
os.environ["TORCH_LOGS"] = "-dynamo"

# Import the original LightningModule
from training.lightning_module import LightningModule

# 1. Load the COCO Mapping
mapping_file = os.path.join("coco-classes-mapping", "coco_mapping_80to91.json")
with open(mapping_file, "r") as f:
    coco_map = {int(k): int(v) for k, v in json.load(f).items()}

bridge = {
    1: 11, 2: 18, 3: 13, 4: 17, 6: 15, 7: 16, 8: 14, 10: 6, 13: 7
}
stuff_bridge = {
    100: 0, 123: 1, 91: 2, 129: 2, 109: 3, 110: 3, 111: 3, 112: 3, 131: 3,
    117: 4, 116: 8, 125: 8, 126: 9, 119: 10
}

def map_to_cityscapes(pred_tensor):
    mapped = torch.full_like(pred_tensor, 19)
    for model_id, coco91_id in coco_map.items():
        if coco91_id in bridge:
            mapped[pred_tensor == (model_id - 1)] = bridge[coco91_id]
    for stuff_id, cs_id in stuff_bridge.items():
        mapped[pred_tensor == stuff_id] = cs_id
    return mapped

# 2. Define our custom test_step
def custom_test_step(self, batch, batch_idx):
    imgs, targets = batch

    # Forward pass (Assuming Panoptic COCO model)
    transformed = self.model.resize_and_pad_imgs_instance_panoptic(imgs)
    m_logits_p, c_logits_p = self.model(transformed)
    m_logits_p = F.interpolate(m_logits_p[-1], self.model.img_size, mode="bilinear")
    m_logits_p = self.model.revert_resize_and_pad_logits_instance_panoptic(m_logits_p, [imgs[0].shape[-2:]])

    # Get predictions
    pred = self.model.to_per_pixel_preds_panoptic(
        m_logits_p, c_logits_p[-1], getattr(self.model, 'stuff_classes', list(range(80, 133))), 0.8, 0.8
    )[0][..., 0]

    # Apply our zero-shot map
    mapped_pred = map_to_cityscapes(pred)

    # Format Ground Truth
    gt = torch.full(imgs[0].shape[-2:], 19, device=self.device)
    for m, l in zip(targets[0]['masks'], targets[0]['labels']):
        gt[m] = l
    gt[gt == 255] = 19

    # Send to the metric calculator
    if hasattr(self, 'test_metric'):
        # Add dummy batch dims
        self.test_metric.addBatch(mapped_pred.unsqueeze(0).unsqueeze(0), gt.unsqueeze(0).unsqueeze(0))

# 3. Monkey-Patch the class
LightningModule.test_step = custom_test_step

# 4. Import and execute the professor's explicit CLI logic
from main import cli_main

if __name__ == '__main__':
    # This directly triggers the exact parsing and linking setup designed by your professor
    cli_main()
