from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch.optim import AdamW

from training.mask_classification_semantic import MaskClassificationSemantic
from training.two_stage_warmup_poly_schedule import TwoStageWarmupPolySchedule


class MaskClassificationLoRA(MaskClassificationSemantic):
    """Fine-tune EoMT on Cityscapes with LoRA adapters on the ViT encoder.

    LoRA is injected after the parent __init__ finishes loading the COCO
    checkpoint, so the pre-trained weights are preserved and only adapter
    matrices (lora_A, lora_B) plus the fresh class/mask heads are trained.
    """

    def __init__(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_target_modules: List[str] = ("qkv", "fc1", "fc2"),
        lora_dropout: float = 0.05,
        trainable_blocks: Optional[Union[List[int], range]] = None,
        **kwargs,
    ):
        # Parent loads the COCO checkpoint and builds the network.
        super().__init__(**kwargs)

        self.save_hyperparameters(ignore=["network"])

        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=list(lora_target_modules),
            lora_dropout=lora_dropout,
            bias="none",
        )
        # Wrap encoder with LoRA adapters
        self.network.encoder = get_peft_model(self.network.encoder, lora_cfg)

        # Apply encapsulated freezing logic
        if trainable_blocks is not None:
            trainable_blocks = list(trainable_blocks)
            for name, param in self.network.encoder.named_parameters():
                idx = _block_idx(name)
                # If it belongs to a block not in our list, freeze it
                if idx is not None and idx not in trainable_blocks:
                    param.requires_grad = False
                # Freeze non-block components (like patch_embed) if they were made trainable by PEFT
                elif idx is None and "lora_" in name:
                    param.requires_grad = False
        
        self.network.encoder.print_trainable_parameters()

    def validation_step(self, batch, batch_idx=0):
        imgs, targets = batch
        batch_size = len(imgs)
        mask_logits_per_block, class_logits_per_block = self(imgs)
        val_loss = None
        for mask_logits, class_logits in zip(mask_logits_per_block, class_logits_per_block):
            losses = self.criterion(
                masks_queries_logits=mask_logits,
                class_queries_logits=class_logits,
                targets=targets,
            )
            for loss_key, loss in losses.items():
                if "mask" in loss_key:
                    w = loss * self.criterion.mask_coefficient
                elif "dice" in loss_key:
                    w = loss * self.criterion.dice_coefficient
                elif "cross_entropy" in loss_key:
                    w = loss * self.criterion.class_coefficient
                else:
                    continue
                val_loss = w if val_loss is None else val_loss + w
        self.log("losses/val_loss_total", val_loss, on_step=False, on_epoch=True,
                 sync_dist=True, prog_bar=True, batch_size=batch_size)

    def on_validation_epoch_end(self):
        pass  # no mIoU during fine-tuning; loss is logged per-step above

    def on_validation_end(self):
        pass

    def configure_optimizers(self):
        """Standardize optimizer setup: LLRD + Weight Decay exclusion for Biases/Norms."""
        num_blocks = len(self.network.encoder.backbone.blocks)
        
        backbone_groups: list[dict] = []
        other_groups: list[dict] = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            lr = self.lr
            is_backbone = "network.encoder" in name
            
            # 1. Apply LLRD for backbone components
            if is_backbone:
                block_idx = _block_idx(name)
                if block_idx is not None:
                    lr *= self.llrd ** (num_blocks - 1 - block_idx)
            
            # 2. Exclude biases and LayerNorm from Weight Decay
            wd = self.weight_decay
            if param.ndim == 1 or "bias" in name or "norm" in name:
                wd = 0.0
            
            group = {"params": [param], "lr": lr, "weight_decay": wd}
            
            if is_backbone:
                backbone_groups.append(group)
            else:
                other_groups.append(group)

        # The scheduler assumes backbone groups come FIRST
        optimizer = AdamW(backbone_groups + other_groups)

        scheduler = TwoStageWarmupPolySchedule(
            optimizer,
            num_backbone_params=len(backbone_groups),
            warmup_steps=self.warmup_steps,
            total_steps=self.trainer.estimated_stepping_batches,
            poly_power=self.poly_power,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


def _block_idx(param_name: str) -> Optional[int]:
    """Extract the transformer block index from a dotted parameter name."""
    parts = param_name.split(".")
    for i, part in enumerate(parts):
        if part == "blocks" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None
