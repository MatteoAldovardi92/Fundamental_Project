import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Cityscapes Classes
cs_classes = {
    0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence',
    5: 'pole', 6: 'traffic light', 7: 'traffic sign', 8: 'vegetation',
    9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car',
    14: 'truck', 15: 'bus', 16: 'train', 17: 'motorcycle', 18: 'bicycle'
}

# 2. Load the official 133 Panoptic COCO categories
# The model output indices (0-132) perfectly match the order of this list!
panoptic_categories_path = '/content/drive/MyDrive/FundGitHubProject/eomt/data/panoptic_coco_categories.json'
with open(panoptic_categories_path, 'r') as f:
    categories = json.load(f)

# 3. Define synonyms for domain bridging
synonyms = {
    'tree': 'vegetation', 'bush': 'vegetation', 'flower': 'vegetation',
    'grass': 'terrain', 'dirt': 'terrain', 'playingfield': 'terrain',
    'mud': 'terrain', 'sand': 'terrain', 'gravel': 'terrain', 
    'hill': 'terrain', 'mountain': 'terrain', 'rock': 'terrain',
    'pavement': 'sidewalk',
    'motorbike': 'motorcycle',
    'stop sign': 'traffic sign', 'street sign': 'traffic sign',
    'house': 'building', 'roof': 'building', 'tent': 'building', 'bridge': 'building',
    'clouds': 'sky', 'fog': 'sky'
}

# 4. Build the correct numerical map (Model Output Index 0-132 -> Cityscapes ID)
numerical_map = {}
for model_idx, cat in enumerate(categories):
    coco_name = cat['name']
    
    # Suffix stripping handles 'wall-brick', 'wall-stone', 'rug-merged', etc.
    clean_name = coco_name.split('-')[0]
    clean_name = synonyms.get(clean_name, clean_name)
    
    for cs_id, cs_name in cs_classes.items():
        if clean_name == cs_name:
            numerical_map[model_idx] = cs_id

# 5. Save the robust mapping
save_path = os.path.join(script_dir, 'coco_to_cs.json')
with open(save_path, 'w') as f:
    json.dump(numerical_map, f)
    
print(f"Dynamically generated robust map with {len(numerical_map)} class connections.")
print(f"Saved to {save_path}")
