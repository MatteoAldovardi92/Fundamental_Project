from typing import List, Optional

from torch.optim import AdamW

from training.mask_classification_semantic import MaskClassificationSemantic
from training.two_stage_warmup_poly_schedule import TwoStageWarmupPolySchedule


class MaskClassificationLoRA(MaskClassificationSemantic):
    """Fine-tune EoMT on Cityscapes with LoRA adapters on the ViT encoder.

    LoRA is injected after the parent __init__ finishes loading the COCO
    checkpoint, so the pre-trained weights are preserved and only adapter
    matrices (lora_A, lora_B) plus the fresh class/mask heads are trained.

    Why override configure_optimizers?
    The parent implementation identifies backbone parameters by checking
    membership in encoder.backbone.named_parameters().  After get_peft_model
    wraps the encoder, lora_A/B matrices are added by PEFT and are NOT in
    that set.  They would silently land in other_param_groups and receive the
    full base LR with no LLRD.  Here we extract the block index directly from
    the parameter name string, which works uniformly for original and
    PEFT-added parameters.
    """

    def __init__(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_target_modules: List[str] = ('qkv', 'fc1', 'fc2'),
        lora_dropout: float = 0.05,
        **kwargs,
    ):
        # Parent loads the COCO checkpoint and builds the network.
        # We inject LoRA afterwards so the adapter matrices are zero-initialised
        # on top of the already-loaded weights.
        super().__init__(**kwargs)

        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=list(lora_target_modules),
            lora_dropout=lora_dropout,
            bias='none',
        )
        self.network.encoder = get_peft_model(self.network.encoder, lora_cfg)
        self.network.encoder.print_trainable_parameters()

    def configure_optimizers(self):
        num_blocks = len(self.network.encoder.backbone.blocks)

        backbone_groups: list[dict] = []
        other_groups:   list[dict] = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            lr = self.lr

            if 'network.encoder' in name:
                block_idx = _block_idx(name)
                if block_idx is not None:
                    lr *= self.llrd ** (num_blocks - 1 - block_idx)
                backbone_groups.append({'params': [param], 'lr': lr})
            else:
                other_groups.append({'params': [param], 'lr': lr})

        optimizer = AdamW(
            backbone_groups + other_groups,
            weight_decay=self.weight_decay,
        )

        scheduler = TwoStageWarmupPolySchedule(
            optimizer,
            num_backbone_params=len(backbone_groups),
            warmup_steps=self.warmup_steps,
            total_steps=self.trainer.estimated_stepping_batches,
            poly_power=self.poly_power,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1},
        }


def _block_idx(param_name: str) -> Optional[int]:
    """Extract the transformer block index from a dotted parameter name."""
    parts = param_name.split('.')
    for i, part in enumerate(parts):
        if part == 'blocks' and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None
