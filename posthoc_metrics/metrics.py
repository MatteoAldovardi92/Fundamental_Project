import numpy as np
from sklearn.metrics import average_precision_score, roc_curve

from sklearn.metrics import average_precision_score, roc_curve
import numpy as np

def compute_metrics(anomaly_scores, ground_truth):
    """
    Computes AUPRC and FPR95 given anomaly scores and binary ground truth.
    
    Args:
        anomaly_scores: Flat array of anomaly scores (higher = more anomalous).
        ground_truth: Flat array of binary ground truth.
                      Convention: 0 = anomaly, 1 = normal (will be inverted internally).
    
    Returns:
        dict with 'auprc' and 'fpr95'.
    """
    mask = (ground_truth == 0) | (ground_truth == 1)
    scores = anomaly_scores[mask]
    labels = ground_truth[mask]
    
    if len(labels) == 0:
        return {'auprc': 0.0, 'fpr95': 1.0}

    # datasets use 0=anomaly, 1=normal — invert to match sklearn convention
    #labels = 1 - labels

    auprc = average_precision_score(labels, scores)
    
    fpr, tpr, _ = roc_curve(labels, scores)
    tpr_mask = tpr >= 0.95
    fpr95 = fpr[tpr_mask][0] if tpr_mask.any() else 1.0
    
    return {'auprc': auprc, 'fpr95': fpr95}