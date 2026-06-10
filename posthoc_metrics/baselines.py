import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# PIXEL-BASED METHODS (e.g., for ERFNet)
# Expected input: logits [B, C, H, W]
# -----------------------------------------------------------------------------

def get_pixel_msp(logits, temperature=1.0):
    """Maximum Softmax Probability (with slicing trick)"""
    # The void/unlabeled class (index 0) was excluded here during early experiments

    # Higher confidence on known classes --> lower anomaly score
    probs = F.softmax(logits / temperature, dim=1)
    max_probs, _ = torch.max(probs, dim=1)
    return 1.0 - max_probs # flip: high confidence = inlier, low confidence = anomaly

def get_pixel_max_logit(logits):
    """Maximum Logit (with slicing trick)"""
    
    # Same idea as MSP but skips the softmax
    max_logits, _ = torch.max(logits, dim=1)
    return -max_logits # negate so anomalies have high scores, same convention as MSP

def get_pixel_entropy(logits, temperature=1.0):
    """Maximum Entropy (with slicing trick)"""
    
    probs = F.softmax(logits / temperature, dim=1)
    # Small epsilon prevents log(0); won't affect results meaningfully
    entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=1)
    num_classes = logits.shape[1]
    # Divide by log(C) to keep scores in [0, 1] regardless of number of classes
    return entropy / torch.log(torch.tensor(float(num_classes)))


# -----------------------------------------------------------------------------
# MASK-BASED METHODS (e.g., for EoMT)
# Expected input: mask_cls [B, Q, C+1], mask_pred [B, Q, H, W]
# -----------------------------------------------------------------------------

def _reconstruct_dense_probs(mask_cls, mask_pred, temperature=1.0):
    """Helper: (B, Q, C) x (B, Q, H, W) -> (B, C, H, W)"""
    
     # mask_cls has C+1 classes; the last one is the "no-object" bin --> drop it
    class_probs = F.softmax(mask_cls / temperature, dim=-1)[..., :-1] # (B, Q, C)
    mask_probs = torch.sigmoid(mask_pred) # (B, Q, H, W)
    
    # Weighted sum: each query contributes to each class at each pixel
    dense = torch.einsum('bnc,bnhw->bchw', class_probs, mask_probs)
    
    # Normalize so per-pixel class probs sum to 1
    dense = dense / (dense.sum(dim=1, keepdim=True) + 1e-7)
    return dense

def get_mask_msp(mask_cls, mask_pred, temperature=1.0):
    """Maximum Softmax Probability (Mask Version)"""
    dense_probs = _reconstruct_dense_probs(mask_cls, mask_pred, temperature)
    max_probs, _ = torch.max(dense_probs, dim=1)
    return 1.0 - max_probs

def get_mask_max_logit(mask_cls, mask_pred):
    """Maximum Logit (Mask Version)"""
    # Skip softmax and work with raw logits
    inlier_logits = mask_cls[..., :-1] # drop the no-object class, same as abov
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

def get_mask_rba(mask_cls, mask_pred, temperature=1.0, method='logsumexp'):
    """
    Region-based Anomaly detection (RbA).
    Two variants: 'tanh' (simpler) and 'logsumexp' (the one from the paper).
    """
    if method == 'tanh':
        # Tanh squashes class probs before summing: anomalous pixels saturate less
        dense_probs = _reconstruct_dense_probs(mask_cls, mask_pred, temperature)
        return -torch.sum(torch.tanh(dense_probs), dim=1)
    elif method == 'logsumexp':
        # Score each query by how well it fits any known class.
        # LogSumExp = soft-max over classes; negate so low inlier confidence = high anomaly score
        inlier_logits = mask_cls[..., :-1]
        mask_probs = torch.sigmoid(mask_pred)
        query_scores = -torch.logsumexp(inlier_logits / temperature, dim=-1)
        
        # Splat query-level scores onto the image,
        #weighted by how much each query owns each pixel
        numerator = torch.einsum('bn,bnhw->bhw', query_scores, mask_probs)
        # Denominator normalizes for varying mask coverage
        denominator = torch.sum(mask_probs, dim=1) + 1e-7
        return numerator / denominator
    else:
        raise ValueError(f"Unknown RbA method: {method}")
