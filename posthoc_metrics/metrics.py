import numpy as np
from sklearn.metrics import average_precision_score, roc_curve

def compute_metrics(anomaly_scores, ground_truth):
    """
    Computes AUPRC and FPR95 given anomaly scores and binary ground truth.
    
    Args:
        anomaly_scores: Flat array of anomaly scores.
        ground_truth: Flat array of binary ground truth (1 for anomaly, 0 for inlier).
        Note: Ignore label is 255.
        
    Returns:
        dict: A dictionary containing 'auprc' and 'fpr95'.
    """
    # 1. Filter out ignore labels (255) and ensure binary labels
    mask = (ground_truth == 0) | (ground_truth == 1)
    scores = anomaly_scores[mask]
    labels = ground_truth[mask]
    
    if len(labels) == 0:
        return {'auprc': 0.0, 'fpr95': 100.0}

    # 2. Compute AuPRC
    auprc = average_precision_score(labels, scores)
    
    # 3. Compute FPR95 (False Positive Rate at 95% TPR)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    # Find the index where TPR is >= 0.95
    idx = np.searchsorted(tpr, 0.95)
    fpr95 = fpr[idx]
    
    return {
        'auprc': auprc,
        'fpr95': fpr95
    }
