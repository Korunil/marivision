"""
datasets/seaclips_detection.py
Clean, standalone SeaClips dataset adapter for official Deformable-DETR.
- Integrates pycocotools.coco.COCO cleanly
- Dynamically creates cat_id_to_label and label_to_cat_id mappings
- Implements 640x640 non-aspect-preserving resize and ImageNet normalization
- Supports deterministic loading (random_flip=False)
"""

from collections import defaultdict
import io
import zipfile
from pathlib import Path
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from pycocotools.coco import COCO

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class SeaClipsDetection(Dataset):
    def __init__(self, img_source_path, ann_file_path, is_train=False, random_flip=False):
        super().__init__()
        self.img_source_path = Path(img_source_path)
        self.ann_file_path = Path(ann_file_path)
        self.is_train = is_train
        self.random_flip = random_flip

        assert self.ann_file_path.exists(), f"Annotation file not found: {self.ann_file_path}"
        self.coco = COCO(str(self.ann_file_path))
        
        # Dynamic, bidirectional category mapping
        self.raw_cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_label = {cat_id: i for i, cat_id in enumerate(self.raw_cat_ids)}
        self.label_to_cat_id = {i: cat_id for i, cat_id in enumerate(self.raw_cat_ids)}

        self.images = self.coco.imgs
        self.img_ids = sorted(list(self.images.keys()))
        self.annotations = self.coco.imgToAnns

        self.zip_file = None
        if self.img_source_path.is_file() and self.img_source_path.suffix.lower() == '.zip':
            self.is_zip = True
        elif self.img_source_path.is_dir():
            self.is_zip = False
        else:
            raise FileNotFoundError(f"Invalid image source path: {self.img_source_path}")

    def _open_zip_if_needed(self):
        if self.zip_file is not None:
            return

        self.zip_file = zipfile.ZipFile(self.img_source_path, "r")
        self.zip_member_map = {}
        for member in self.zip_file.namelist():
            norm = member.replace("\\", "/").strip("/")
            if norm in self.zip_member_map:
                raise RuntimeError(f"Duplicate exact ZIP path: {norm}")
            self.zip_member_map[norm] = member

    def _resolve_zip_member(self, file_name):
        self._open_zip_if_needed()
        fn = file_name.replace("\\", "/").strip("/")
        ann_name = self.ann_file_path.name.lower()

        if "train" in ann_name:
            split = "train"
        elif "val" in ann_name:
            split = "val"
        elif "test" in ann_name:
            split = "test"
        else:
            raise RuntimeError(
                f"Cannot infer SeaClips split from annotation file: {self.ann_file_path}"
            )

        expected = f"SeaClips/{split}/{fn}"
        if expected not in self.zip_member_map:
            raise FileNotFoundError(
                f"Missing exact SeaClips member: {expected}"
            )

        return self.zip_member_map[expected]

    def _load_image(self, img_info):
        file_name = img_info["file_name"]
        if self.is_zip:
            member = self._resolve_zip_member(file_name)
            raw_bytes = self.zip_file.read(member)
            return Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        img_path = self.img_source_path / file_name
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")
        return Image.open(img_path).convert("RGB")


    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        img = self._load_image(img_info)
        orig_w, orig_h = img.size

        anns = self.annotations.get(img_id, [])
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            cx = (x + w / 2.0) / orig_w
            cy = (y + h / 2.0) / orig_h
            nw = w / orig_w
            nh = h / orig_h
            boxes.append([cx, cy, nw, nh])
            labels.append(self.cat_id_to_label[ann['category_id']])

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        img_resized = TF.resize(img, [640, 640])

        if self.random_flip and torch.rand(1).item() > 0.5:
            img_resized = TF.hflip(img_resized)
            if len(boxes) > 0:
                boxes[:, 0] = 1.0 - boxes[:, 0]

        img_tensor = TF.to_tensor(img_resized)
        img_norm = TF.normalize(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'orig_size': torch.tensor([orig_h, orig_w]),
            'size': torch.tensor([640, 640])
        }
        return img_norm, target