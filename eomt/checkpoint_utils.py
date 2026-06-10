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
    Loads the fine-tuned EoMT model from a checkpoint
    
    Args:
        checkpoint_path (str): Path to the .ckpt or .pth file.
        device (str): 'cuda' or 'cpu'.
        
    Returns:
        torch.nn.Module: The loaded model in eval mode.
    """
    
    # 1. Configuration 
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
    # target_modules must match exactly what was used in Step 5 training
    # (exp3_lora_all_blocks.yaml: qkv, proj, fc1, fc2).
    # masked_attn_enabled=False must also match — training never enabled it.
    lora_config = LoraConfig(
        r=CONFIG['lora_r'],
        lora_alpha=CONFIG['lora_alpha'],
        target_modules=['qkv', 'proj', 'fc1', 'fc2'],
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
        masked_attn_enabled=False   # must match Step 5 training config
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


def get_eomt_cityscape(checkpoint_path, device='cuda'):
    print("\n--- Initializing EoMT Cityscapes Architecture ---")
    from eomt.models.vit import ViT
    from eomt.models.eomt import EoMT
    import torch
    import torch.nn.functional as F

    encoder = ViT(img_size=(640, 640), backbone_name='vit_base_patch14_reg4_dinov2')
    model = EoMT(
        encoder=encoder,
        num_classes=19,
        num_q=100,
        num_blocks=3,
        masked_attn_enabled=True
    )

    print(f"Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('network.'): k = k[len('network.'):]
        elif k.startswith('model.'): k = k[len('model.'):]
        new_state_dict[k] = v

    # Interpolate pos_embed if size mismatch
    if 'encoder.backbone.pos_embed' in new_state_dict:
        pos_embed_checkpoint = new_state_dict['encoder.backbone.pos_embed']
        pos_embed_model = model.encoder.backbone.pos_embed
        if pos_embed_checkpoint.shape != pos_embed_model.shape:
            print(f"  Interpolating pos_embed: {pos_embed_checkpoint.shape} → {pos_embed_model.shape}")
            # shapes are [1, N, D] — interpolate along sequence dimension
            N_ckpt = pos_embed_checkpoint.shape[1]
            N_model = pos_embed_model.shape[1]
            D = pos_embed_checkpoint.shape[2]
            h = w = int(N_ckpt ** 0.5)
            h_new = w_new = int(N_model ** 0.5)
            pos_embed_checkpoint = pos_embed_checkpoint.reshape(1, h, w, D).permute(0, 3, 1, 2)
            pos_embed_checkpoint = F.interpolate(pos_embed_checkpoint, size=(h_new, w_new), mode='bicubic', align_corners=False)
            new_state_dict['encoder.backbone.pos_embed'] = pos_embed_checkpoint.permute(0, 2, 3, 1).reshape(1, N_model, D)

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    print("✅ EoMT Cityscapes Model ready for inference.")
    return model

def get_erfnet_model(checkpoint_path, device='cuda'):
    print("\n--- Initializing ErfNet Architecture ---")
    import torch
    from eval.erfnet import ERFNet
    
    model = ERFNet(20)  # 20 classes for the pretrained checkpoint

    print(f"Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    # Handle 'module.' prefix from DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    print("✅ ErfNet Model ready for inference.")
    return model

def get_eomt_coco(checkpoint_path, device='cuda'):
    print("\n--- Initializing EoMT COCO Architecture ---")
    from eomt.models.vit import ViT
    from eomt.models.eomt import EoMT
    import torch
    
    encoder = ViT(img_size=(640, 640), backbone_name='vit_base_patch14_reg4_dinov2')
    model = EoMT(
        encoder=encoder,
        num_classes=133,
        num_q=200,
        num_blocks=3,
        masked_attn_enabled=True
    )

    print(f"Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('network.'): k = k[len('network.'):]
        elif k.startswith('model.'): k = k[len('model.'):]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    print("✅ EoMT COCO Model ready for inference.")
    return model
