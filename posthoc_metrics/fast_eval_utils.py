import torch
import os
from tqdm import tqdm

def cache_model_outputs(model, dataloader, cache_dir, device='cuda'):
    """
    Step 1: Run inference ONCE and save raw logits to disk.
    This is the 'Pro Tip' to save hours of computation.
    """
    os.makedirs(cache_dir, exist_ok=True)
    model.eval()
    model.to(device)
    
    print(f"--- Caching outputs to {cache_dir} ---")
    with torch.no_grad():
        for i, (img, target) in enumerate(tqdm(dataloader)):
            img = img.to(device)
            # Ensure img has batch dimension
            if img.dim() == 3: img = img.unsqueeze(0)
            
            mask_logits_per_layer, class_logits_per_layer = model(img)
            
            # We only need the last layer for post-hoc metrics
            data = {
                'mask_pred': mask_logits_per_layer[-1].cpu(),
                'mask_cls': class_logits_per_layer[-1].cpu(),
                'gt': target.cpu() if isinstance(target, torch.Tensor) else target
            }
            
            torch.save(data, os.path.join(cache_dir, f"sample_{i}.pt"))

def fast_temperature_search(cache_dir, scoring_fn=None, temperatures=[0.5, 0.75, 1.0, 1.1, 1.5]):
    from posthoc_metrics import get_mask_msp, compute_metrics
    import numpy as np

    if scoring_fn is None:
        scoring_fn = get_mask_msp

    results = {}
    cache_files = sorted(f for f in os.listdir(cache_dir) if f.endswith('.pt'))

    for T in temperatures:
        print(f"Testing Temperature T={T}...")
        all_scores = []
        all_gts = []

        for file in cache_files:
            data = torch.load(os.path.join(cache_dir, file), map_location='cpu')
            anomaly_map = scoring_fn(data['mask_cls'], data['mask_pred'], temperature=T)
            gt = data['gt']

            # Upsample score to GT resolution if needed
            if anomaly_map.shape[-2:] != gt.shape[-2:]:
                anomaly_map = torch.nn.functional.interpolate(
                    anomaly_map.unsqueeze(1),
                    size=gt.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)

            all_scores.append(anomaly_map.flatten().numpy())
            all_gts.append(gt.flatten().numpy())

        metrics = compute_metrics(np.concatenate(all_scores), np.concatenate(all_gts))
        print(f"  T={T} -> AuPRC: {metrics['auprc']:.4f} | FPR95: {metrics['fpr95']:.4f}")
        results[T] = metrics

    return results