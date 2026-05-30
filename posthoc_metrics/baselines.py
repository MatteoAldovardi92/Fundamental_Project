import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# PIXEL-BASED METHODS (e.g., for ERFNet)
# Expected input: logits [B, C, H, W]
# -----------------------------------------------------------------------------

def get_pixel_msp(logits, temperature=1.0):
    """Maximum Softmax Probability (with slicing trick)"""
    # If 20 channels, slice to 19 to ignore the 'Void' class during OOD detection
    if logits.shape[1] == 20:
        logits = logits[:, :19, :, :]
    probs = F.softmax(logits / temperature, dim=1)
    max_probs, _ = torch.max(probs, dim=1)
    return 1.0 - max_probs

def get_pixel_max_logit(logits):
    """Maximum Logit (with slicing trick)"""
    if logits.shape[1] == 20:
        logits = logits[:, :19, :, :]
    max_logits, _ = torch.max(logits, dim=1)
    return -max_logits

def get_pixel_entropy(logits, temperature=1.0):
    """Maximum Entropy (with slicing trick)"""
    if logits.shape[1] == 20:
        logits = logits[:, :19, :, :]
    probs = F.softmax(logits / temperature, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=1)
    num_classes = logits.shape[1]
    return entropy / torch.log(torch.tensor(float(num_classes)))


# -----------------------------------------------------------------------------
# MASK-BASED METHODS (e.g., for EoMT)
# Expected input: mask_cls [B, Q, C+1], mask_pred [B, Q, H, W]
# -----------------------------------------------------------------------------

def _reconstruct_dense_probs(mask_cls, mask_pred, temperature=1.0):
    """Helper: (B, Q, C) x (B, Q, H, W) -> (B, C, H, W)"""
    # Drop the 'No-Object' class (last index)
    class_probs = F.softmax(mask_cls / temperature, dim=-1)[..., :-1]
    mask_probs = torch.sigmoid(mask_pred)
    return torch.einsum('bnc,bnhw->bchw', class_probs, mask_probs)

def get_mask_msp(mask_cls, mask_pred, temperature=1.0):
    """Maximum Softmax Probability (Mask Version)"""
    dense_probs = _reconstruct_dense_probs(mask_cls, mask_pred, temperature)
    max_probs, _ = torch.max(dense_probs, dim=1)
    return 1.0 - max_probs

def get_mask_max_logit(mask_cls, mask_pred):
    """Maximum Logit (Mask Version)"""
    # Use raw inlier logits for reconstruction
    inlier_logits = mask_cls[..., :-1]
    mask_probs = torch.sigmoid(mask_pred)
    dense_logits = torch.einsum('bnc,bnhw->bchw', inlier_logits, mask_probs)
    max_logits, _ = torch.max(dense_logits, dim=1)
    return -max_logits

def get_mask_entropy(mask_cls, mask_pred, temperature=1.0):
    """Maximum Entropy (Mask Version)"""
    dense_probs = _reconstruct_dense_probs(mask_cls, mask_pred, temperature)
    entropy = -torch.sum(dense_probs * torch.log(dense_probs + 1e-7), dim=1)
    num_classes = dense_probs.shape[1]
    return entropy / torch.log(torch.tensor(float(num_classes)))

def get_mask_rba(mask_cls, mask_pred, temperature=1.0, method='tanh'):
    """Region-based Anomaly detection (RbA)"""
    if method == 'tanh':
        dense_probs = _reconstruct_dense_probs(mask_cls, mask_pred, temperature)
        return -torch.sum(torch.tanh(dense_probs), dim=1)
    elif method == 'logsumexp':
        inlier_logits = mask_cls[..., :-1]
        mask_probs = torch.sigmoid(mask_pred)
        query_scores = -torch.logsumexp(inlier_logits / temperature, dim=-1)
        numerator = torch.einsum('bn,bnhw->bhw', query_scores, mask_probs)
        denominator = torch.sum(mask_probs, dim=1) + 1e-7
        return numerator / denominator
    else:
        raise ValueError(f"Unknown RbA method: {method}")
