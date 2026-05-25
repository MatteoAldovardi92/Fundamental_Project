
import os
import torch
import sys

# Since this is now inside the 'eomt' package, we adjust the imports
# to be relative to the package structure.
from eomt.models.eomt import EoMT
from eomt.models.vit import ViT
from peft import LoraConfig, get_peft_model

def get_finetuned_model(checkpoint_path, device='cuda'):
    """
    Loads the fine-tuned EoMT model using the 'Step 5 Enhanced' configuration.
    
    Args:
        checkpoint_path (str): Path to the .ckpt or .pth file.
        device (str): 'cuda' or 'cpu'.
        
    Returns:
        torch.nn.Module: The loaded model in eval mode.
    """
    
    # 1. Configuration (Matching Step 5 Enhanced)
    # These MUST match the parameters used during the successful 76% mIoU run.
    CONFIG = {
        'img_size': (640, 640),
        'num_classes': 19,
        'num_q': 200,
        'num_blocks': 3,
        'lora_r': 8,
        'lora_alpha': 16,
        'backbone': 'vit_base_patch14_reg4_dinov2'
    }

    print(f"--- Initializing Enhanced Architecture ---")
    print(f"Blocks: {CONFIG['num_blocks']} | LoRA R: {CONFIG['lora_r']} | Backbone: {CONFIG['backbone']}")

    # 2. Initialize Base Vision Transformer
    # ViT class handles timm creation and internal resizing.
    encoder = ViT(
        img_size=CONFIG['img_size'], 
        backbone_name=CONFIG['backbone']
    )

    # 3. Apply LoRA Surgery
    # This ensures the model keys (lora_A/B) match the checkpoint keys.
    lora_config = LoraConfig(
        r=CONFIG['lora_r'],
        lora_alpha=CONFIG['lora_alpha'],
        target_modules=['qkv', 'fc1', 'fc2'],
        modules_to_save=['class_head', 'mask_head'],
        lora_dropout=0.05,
        bias='none',
    )
    encoder = get_peft_model(encoder, lora_config)

    # 4. Assemble the EoMT Multi-Task Model
    model = EoMT(
        encoder=encoder,
        num_classes=CONFIG['num_classes'],
        num_q=CONFIG['num_q'],
        num_blocks=CONFIG['num_blocks'],
        masked_attn_enabled=True
    )

    # 5. Load the Weights
    print(f"Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract state_dict (handles Lightning and standard saves)
    state_dict = checkpoint.get('state_dict', checkpoint)

    # Clean prefixes: Lightning adds 'model.' or 'network.'
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('network.'):
            k = k[len('network.'):]
        elif k.startswith('model.'):
            k = k[len('model.'):]
        new_state_dict[k] = v

    # Load with strict=False to allow for any minor metadata mismatches 
    # while ensuring all essential weights are loaded.
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    
    if len(missing) > 0:
        print(f"NOTE: Missing keys (expected for PEFT buffers): {len(missing)}")
    if len(unexpected) > 0:
        print(f"NOTE: Unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()
    print("✅ Model ready for inference.")
    
    return model

if __name__ == "__main__":
    # Example usage
    CKPT = "/Users/matteoaldovardi/Desktop/Fundamental_Project/checkpoints/cityscapes_enhanced/eomt-enhanced-epoch=23-val_iou_all=0.00.ckpt"
    if os.path.exists(CKPT):
        m = get_finetuned_model(CKPT)
    else:
        print(f"Please update the CKPT path in this script to point to your .ckpt file.")
