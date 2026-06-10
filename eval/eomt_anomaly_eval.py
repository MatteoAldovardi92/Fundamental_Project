import torch
import torch.nn.functional as F
from eomt.training.lightning_module import LightningModule
from RbA_main.support import get_datasets, OODEvaluator

# 1. Load the model using the existing Lightning wrapper
# You need the config.yaml and the .ckpt file
model = LightningModule.load_from_checkpoint(checkpoint_path, config=cfg)
model.eval().cuda()

def eomt_rba_scoring(model, images, temperature=1.0):
   # images should be in [0, 255] as expected by LightningModule.forward
   with torch.no_grad():
      # Get mask and class logits from the last layer
      mask_logits_list, class_logits_list = model.network(images / 255.0)
      mask_logits = mask_logits_list[-1]
      class_logits = class_logits_list[-1]
   
      # 2. Apply Temperature Scaling to class logits
      scaled_class_logits = class_logits / temperature
      
      # 3. Construct Semantic Map (Per-pixel class probabilities)
      mask_probs = mask_logits.sigmoid() # (B, Q, H, W)
      class_probs = F.softmax(scaled_class_logits, dim=-1)[..., :-1] #(B, Q, C)
            
      # Combine masks and classes
      semseg = torch.einsum("bqhw, bqc -> bchw", mask_probs, class_probs)
   
      # 4. Apply RbA Scoring
      # The anomaly score is the negative sum of tanh of the class probabilities
      anomaly_score = -torch.tanh(semseg).sum(dim=1) # (B, H, W)
            
   return anomaly_score
   
# 5. Use the OOD Evaluator from the RbA module
datasets = get_datasets(datasets_root)
evaluator = OODEvaluator(model, inference_func=None, anomaly_score_func=eomt_rba_scoring)

# Run evaluation on RoadAnomaly or Fishyscapes
metrics = evaluator.evaluate_ood(anomaly_score, ood_gts)