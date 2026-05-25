    1 import torch
    2 import torch.nn.functional as F
    3 from eomt.training.lightning_module import LightningModule
    4 from RbA_main.support import get_datasets, OODEvaluator
    5
    6 # 1. Load the model using the existing Lightning wrapper
    7 # You need the config.yaml and the .ckpt file
    8 model = LightningModule.load_from_checkpoint(checkpoint_path, config=cfg)
    9 model.eval().cuda()
   10
   11 def eomt_rba_scoring(model, images, temperature=1.0):
   12     # images should be in [0, 255] as expected by LightningModule.forward
   13     with torch.no_grad():
   14         # Get mask and class logits from the last layer
   15         mask_logits_list, class_logits_list = model.network(images /
      255.0)
   16         mask_logits = mask_logits_list[-1]
   17         class_logits = class_logits_list[-1]
   18
   19         # 2. Apply Temperature Scaling to class logits
   20         scaled_class_logits = class_logits / temperature
   21
   22         # 3. Construct Semantic Map (Per-pixel class probabilities)
   23         mask_probs = mask_logits.sigmoid() # (B, Q, H, W)
   24         class_probs = F.softmax(scaled_class_logits, dim=-1)[..., :-1] #
      (B, Q, C)
   25         
   26         # Combine masks and classes
   27         semseg = torch.einsum("bqhw, bqc -> bchw", mask_probs,
      class_probs)
   28
   29         # 4. Apply RbA Scoring
   30         # The anomaly score is the negative sum of tanh of the class
      probabilities
   31         anomaly_score = -torch.tanh(semseg).sum(dim=1) # (B, H, W)
   32         
   33     return anomaly_score
   34
   35 # 5. Use the OOD Evaluator from the RbA module
   36 datasets = get_datasets(datasets_root)
   37 evaluator = OODEvaluator(model, inference_func=None,
      anomaly_score_func=eomt_rba_scoring)
   38
   39 # Run evaluation on RoadAnomaly or Fishyscapes
   40 metrics = evaluator.evaluate_ood(anomaly_score, ood_gts)