import torch
import torch.nn.functional as F

def get_dense_probs(mask_cls, mask_pred, temperature=1.0):
    """
    Reconstructs the dense probability map from query classes and masks.
    """
    class_probs = F.softmax(mask_cls / temperature, dim=-1)[..., :-1]
    mask_probs = torch.sigmoid(mask_pred)
    # (Batch, Class, H, W)
    return torch.einsum('bnc,bnhw->bchw', class_probs, mask_probs)

def get_msp_anomaly_map(mask_cls, mask_pred, temperature=1.0):
    """
    Maximum Softmax Probability (MSP)
    Anomaly score = 1.0 - max(class_probability)
    """
    dense_probs = get_dense_probs(mask_cls, mask_pred, temperature)
    max_probs, _ = torch.max(dense_probs, dim=1)
    return 1.0 - max_probs

def get_max_logit_anomaly_map(mask_cls, mask_pred, temperature=1.0):
    """
    Max Logit
    Anomaly score = -max(class_logit)
    """
    # For Max Logit, we reconstruct the dense logits (before softmax/sigmoid)
    # This is an approximation for Mask Architectures
    class_logits = mask_cls[..., :-1]
    dense_logits = torch.einsum('bnc,bnhw->bchw', class_logits, mask_pred)
    max_logits, _ = torch.max(dense_logits, dim=1)
    return -max_logits

def get_max_entropy_anomaly_map(mask_cls, mask_pred, temperature=1.0):
    """
    Max Entropy
    Anomaly score = -sum(p * log(p))
    """
    dense_probs = get_dense_probs(mask_cls, mask_pred, temperature)
    # Add a small epsilon to avoid log(0)
    entropy = -torch.sum(dense_probs * torch.log(dense_probs + 1e-7), dim=1)
    return entropy

