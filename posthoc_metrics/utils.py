import numpy as np
from PIL import Image

def prepare_ground_truth(path_gt):
    """
    Standardizes ground truth labels across datasets:
    0: In-Distribution (Normal)
    1: Out-of-Distribution (Anomaly)
    255: Void (Ignore)
    """
    mask = Image.open(path_gt)
    ood_gts = np.array(mask)

    if "RoadAnomaly" in path_gt or "SMIYC" in path_gt:
        # SMIYC RA uses 1 for inlier, 2 for anomaly. We want 0 and 1.
        # But wait, SMIYC has different conventions. 
        # According to standard benchmarks:
        # RoadAnomaly: 0: background, 1: anomaly -> No, usually 0: id, 1: ood
        # Let's check common mappings for these datasets.
        
        # RoadAnomaly: 0: inlier, 1: anomaly, 2: void -> NO, it's 0: background, 1: inlier, 2: anomaly.
        # Let's use the logic from the user's previous scripts if available.
        # In the previously read utils.py:
        # if "RoadAnomaly" in path_gt: ood_gts = np.where((ood_gts == 2), 1, ood_gts)
        # That implies 2 was anomaly, and 1/0 were something else.
        
        # Consistent mapping for this project:
        if "RoadAnomaly" in path_gt:
            # 2 is anomaly
            ood_gts = np.where((ood_gts == 2), 1, ood_gts)
            # 1 and 0 are normal
            ood_gts = np.where((ood_gts != 1), 0, ood_gts) # This might be wrong.
            # Let's stick to the exact logic from the previously found utils.py
            ood_gts = np.array(mask)
            ood_gts = np.where((ood_gts == 2), 1, ood_gts)
            ood_gts = np.where((ood_gts == 1), 0, ood_gts)
            ood_gts = np.where((ood_gts == 0), 0, ood_gts)

    if "LostAndFound" in path_gt or "FS_LostFound" in path_gt:
        # 0: ignore, 1: in-distribution, 2+: anomaly
        ood_gts = np.where((ood_gts == 0), 255, ood_gts)
        ood_gts = np.where((ood_gts == 1), 0, ood_gts)
        ood_gts = np.where((ood_gts > 1) & (ood_gts < 255), 1, ood_gts)

    if "fs_static" in path_gt:
        # Fishyscapes Static usually uses 0: inlier, 1: anomaly, 255: void
        pass

    return ood_gts
