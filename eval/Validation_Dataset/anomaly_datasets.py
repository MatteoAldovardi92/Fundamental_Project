import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import lightning as L
from torchvision.transforms import Compose, Resize, ToTensor

class BaseAnomalyDataset(Dataset):
    """
    Generic dataset for Anomaly validation.
    Centralizes path management and dataset-specific label mapping.
    """
    def __init__(self, root, img_size=(640, 640), transform=None):
        self.root = root
        self.img_size = img_size
        self.transform = transform or Compose([Resize(img_size), ToTensor()])
        
        # Discover images - assumes standard 'images' subfolder
        self.image_paths = sorted(glob.glob(os.path.join(root, "images", "*")))
        if not self.image_paths:
            print(f"Warning: No images found in {os.path.join(root, 'images')}")
        
    def __len__(self):
        return len(self.image_paths)

    def _get_label_path(self, img_path):
        # Metadata logic: handles the various extensions and subfolder names
        label_path = img_path.replace("images", "labels_masks")
        # Try to match the .png mask even if the image is .jpg or .webp
        base = os.path.splitext(label_path)[0]
        return base + ".png"

    def _map_labels(self, label_np, path):
        """
        Metadata Logic: Centralized label mapping rules from evalAnomaly.py
        """
        # RoadAnomaly / RoadAnomaly21 / RoadObstacle21
        if "RoadAnomaly" in path or "RoadObstacle" in path:
            return np.where(label_np == 2, 1, label_np)
        
        # Fishyscapes Lost & Found
        if "LostFound" in path or "LostAndFound" in path:
            return label_np

        # StreetHazard (if applicable)
        if "Streethazard" in path:
            mapped = np.where(label_np == 14, 255, label_np)
            mapped = np.where(mapped < 20, 0, mapped)
            mapped = np.where(mapped == 255, 1, mapped)
            return mapped

        return label_np

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label_path = self._get_label_path(img_path)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # Load label/mask
        try:
            label = Image.open(label_path)
            # Resize mask with NEAREST to preserve labels
            label_np = np.array(label.resize(self.img_size[::-1], Image.NEAREST))
            label_np = self._map_labels(label_np, img_path)
        except FileNotFoundError:
            # Fallback for datasets where masks might be missing or named differently
            label_np = np.zeros((self.img_size[0], self.img_size[1]))

        return {
            'image': image,
            'label': torch.from_numpy(label_np).long(),
            'path': img_path
        }

class AnomalyDataModule(L.LightningDataModule):
    """
    Lightning DataModule that encapsulates all Anomaly Validation datasets.
    Keeps the 'annoying' default paths and settings in one place.
    """
    def __init__(self, dataset_name, root_dir=None, batch_size=1, img_size=(640, 640)):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.img_size = img_size
        
        # Centralized Metadata: Default paths relative to project root
        # Using 'Validation_Dataset' as requested.
        default_roots = {
            'RoadAnomaly': 'eval/Validation_Dataset/RoadAnomaly',
            'RoadAnomaly21': 'eval/Validation_Dataset/RoadAnomaly21',
            'RoadObstacle21': 'eval/Validation_Dataset/RoadObstacle21',
            'FS_Static': 'eval/Validation_Dataset/fs_static',
            'FS_LostFound': 'eval/Validation_Dataset/FS_LostFound_full'
        }
        
        self.root = root_dir or default_roots.get(dataset_name)
        if not self.root:
            raise ValueError(f"Dataset '{dataset_name}' not recognized. Available: {list(default_roots.keys())}")

    def setup(self, stage=None):
        # Resolve path relative to project root if needed
        abs_root = os.path.abspath(self.root)
        self.dataset = BaseAnomalyDataset(abs_root, img_size=self.img_size)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=2, shuffle=False)

    def test_dataloader(self):
        return self.val_dataloader()

class RoadAnomalyDM(AnomalyDataModule):
    def __init__(self, **kwargs):
        super().__init__(dataset_name='RoadAnomaly', **kwargs)

class FSStaticDM(AnomalyDataModule):
    def __init__(self, **kwargs):
        super().__init__(dataset_name='FS_Static', **kwargs)

class FSLostFoundDM(AnomalyDataModule):
    def __init__(self, **kwargs):
        super().__init__(dataset_name='FS_LostFound', **kwargs)

