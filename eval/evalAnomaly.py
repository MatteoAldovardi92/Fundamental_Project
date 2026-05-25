# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
import sys
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch.nn.functional as F

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        # Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--method', default='maxlogits', choices=['maxlogits', 'msp', 'max_entropy'], 
                        help="Anomaly scoring method: maxlogits, msp, or max_entropy")
    args = parser.parse_args()
    
    # --- Setup Structured Logging ---
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(args.loadWeights).replace('.pth', '')
    experiment_id = f"{model_name}_{args.method}_{timestamp}"
    results_dir = os.path.join(os.path.dirname(__file__), 'results_anomaly', experiment_id)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save parameters to config.json
    import json
    with open(os.path.join(results_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    # --------------------------------

    anomaly_score_list = []
    ood_gts_list = []

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)
    print ("Scoring method: " + args.method)
    print (f"Results will be saved to: {results_dir}")

    model = ERFNet(NUM_CLASSES)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print ("Model and weights LOADED successfully")
    model.eval()
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        img = Image.open(path).convert('RGB')
        images = input_transform(img).unsqueeze(0).float().cuda()
        
        with torch.no_grad():
            # SINGLE FORWARD PASS
            logits = model(images)
            
            if args.method == 'maxlogits':
                # Original maxlogits logic: 1 - max(logits)
                anomaly_result = 1.0 - torch.max(logits, dim=1)[0].cpu().numpy()[0]
                
            elif args.method == 'msp':
                #MSP logic: 1 - max(softmax)
                probs = F.softmax(logits, dim=1)
                anomaly_result = 1.0 - torch.max(probs, dim=1)[0].cpu().numpy()[0]
                
            elif args.method == 'max_entropy':
                # Entropy logic: Normalized entropy from baselines (taken from utilities.py)
                probs = F.softmax(logits, dim=1)
                entropy = torch.div(
                    torch.sum(-probs * torch.log(probs + 1e-10), dim=1), 
                    torch.log(torch.tensor(probs.shape[1], dtype=torch.float))
                )
                anomaly_result = entropy.cpu().numpy()[0]
            else:
                raise ValueError(f"Unknown method {args.method}")
              
        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")  

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue              
        else:
             ood_gts_list.append(ood_gts)
             anomaly_score_list.append(anomaly_result)
        if 'result' in locals(): del result
        if 'anomaly_result' in locals(): del anomaly_result
        if 'ood_gts' in locals(): del ood_gts
        if 'mask' in locals(): del mask
        torch.cuda.empty_cache()

    

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
    
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')

    with open(os.path.join(results_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Experiment ID: {experiment_id}\n")
        f.write(f"Scoring Method: {args.method}\n")
        f.write(f"AUPRC score: {prc_auc*100.0}\n")
        f.write(f"FPR@TPR95: {fpr*100.0}\n")
    
    print(f"Metrics successfully saved to {os.path.join(results_dir, 'metrics.txt')}")

if __name__ == '__main__':
    main()
