import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from configs.config import DATA_ROOT, PRIORS_ROOT, IMG_SIZE
from src.heatmap_features import STRUCT_FEAT_DIM, extract_heatmap_struct_features

# ======================================================================
# 3A. Dataset: original image-only path from code 2
# Return: img1, img2, hm1, hm2, label
# ======================================================================
class TwoViewSpineDataset_ImageOnly(Dataset):
    def __init__(self, samples, use_prior=True, train=False):
        self.samples = samples
        self.use_prior = use_prior
        self.train = train
        self.zero_hm = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def _read_gray(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return img.astype(np.float32) / 255.0

    def _read_hm(self, path):
        if path is None:
            return self.zero_hm
        hm = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if hm is None:
            return self.zero_hm
        hm = cv2.resize(hm, (IMG_SIZE, IMG_SIZE))
        return hm.astype(np.float32) / 255.0

    def _augment_pair(self, img, hm):
        # 1) Mild rotation and scaling
        if random.random() < 0.7:
            angle = random.uniform(-8.0, 8.0)
            scale = random.uniform(0.95, 1.05)
            center = (IMG_SIZE / 2, IMG_SIZE / 2)
            M = cv2.getRotationMatrix2D(center, angle, scale)

            img = cv2.warpAffine(
                img, M, (IMG_SIZE, IMG_SIZE),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            hm = cv2.warpAffine(
                hm, M, (IMG_SIZE, IMG_SIZE),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

        # 2) Mild translation
        if random.random() < 0.5:
            tx = random.uniform(-8, 8)
            ty = random.uniform(-8, 8)
            M = np.float32([[1, 0, tx], [0, 1, ty]])

            img = cv2.warpAffine(
                img, M, (IMG_SIZE, IMG_SIZE),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            hm = cv2.warpAffine(
                hm, M, (IMG_SIZE, IMG_SIZE),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

        # 3) Brightness and contrast perturbation, applied only to the image
        if random.random() < 0.7:
            alpha = random.uniform(0.9, 1.1)
            beta = random.uniform(-0.08, 0.08)
            img = np.clip(img * alpha + beta, 0.0, 1.0)

        # 4) Mild Gaussian noise, applied only to the image
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.015, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0.0, 1.0)

        return img.astype(np.float32), hm.astype(np.float32)

    def __getitem__(self, idx):
        p1, p2, h1, h2, label = self.samples[idx]

        img1 = self._read_gray(p1)
        img2 = self._read_gray(p2)

        if self.use_prior:
            hm1 = self._read_hm(h1)
            hm2 = self._read_hm(h2)
        else:
            hm1 = self.zero_hm
            hm2 = self.zero_hm

        if self.train:
            img1, hm1 = self._augment_pair(img1, hm1)
            img2, hm2 = self._augment_pair(img2, hm2)

        img1 = torch.from_numpy(img1).unsqueeze(0).repeat(3, 1, 1)
        img2 = torch.from_numpy(img2).unsqueeze(0).repeat(3, 1, 1)
        hm1  = torch.from_numpy(hm1).unsqueeze(0)
        hm2  = torch.from_numpy(hm2).unsqueeze(0)

        return img1, img2, hm1, hm2, torch.tensor(label, dtype=torch.long)

# ======================================================================
# 3B. Dataset: structure-enhanced path
# Return: img1, img2, hm1, hm2, struct1, struct2, label
# ======================================================================
class TwoViewSpineDataset_WithStruct(Dataset):
    def __init__(self, samples, use_prior=True, train=False):
        self.samples = samples
        self.use_prior = use_prior
        self.train = train
        self.zero_hm = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        self.zero_struct = np.zeros((STRUCT_FEAT_DIM,), dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def _read_gray(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        return img.astype(np.float32) / 255.0

    def _read_hm(self, path):
        if path is None:
            return self.zero_hm.copy()
        hm = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if hm is None:
            return self.zero_hm.copy()
        hm = cv2.resize(hm, (IMG_SIZE, IMG_SIZE))
        return hm.astype(np.float32) / 255.0

    def _augment_pair(self, img, hm):
        # 1) Mild rotation and scaling
        if random.random() < 0.7:
            angle = random.uniform(-8.0, 8.0)
            scale = random.uniform(0.95, 1.05)
            center = (IMG_SIZE / 2, IMG_SIZE / 2)
            M = cv2.getRotationMatrix2D(center, angle, scale)

            img = cv2.warpAffine(
                img, M, (IMG_SIZE, IMG_SIZE),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            hm = cv2.warpAffine(
                hm, M, (IMG_SIZE, IMG_SIZE),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

        # 2) Mild translation
        if random.random() < 0.5:
            tx = random.uniform(-8, 8)
            ty = random.uniform(-8, 8)
            M = np.float32([[1, 0, tx], [0, 1, ty]])

            img = cv2.warpAffine(
                img, M, (IMG_SIZE, IMG_SIZE),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            hm = cv2.warpAffine(
                hm, M, (IMG_SIZE, IMG_SIZE),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

        # 3) Brightness and contrast perturbation, applied only to the image
        if random.random() < 0.7:
            alpha = random.uniform(0.9, 1.1)
            beta = random.uniform(-0.08, 0.08)
            img = np.clip(img * alpha + beta, 0.0, 1.0)

        # 4) Mild Gaussian noise, applied only to the image
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.015, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0.0, 1.0)

        return img.astype(np.float32), hm.astype(np.float32)

    def __getitem__(self, idx):
        p1, p2, h1, h2, label = self.samples[idx]

        img1 = self._read_gray(p1)
        img2 = self._read_gray(p2)

        if self.use_prior:
            hm1 = self._read_hm(h1)
            hm2 = self._read_hm(h2)
        else:
            hm1 = self.zero_hm.copy()
            hm2 = self.zero_hm.copy()

        if self.train:
            img1, hm1 = self._augment_pair(img1, hm1)
            img2, hm2 = self._augment_pair(img2, hm2)

        if self.use_prior:
            struct1 = extract_heatmap_struct_features(hm1)
            struct2 = extract_heatmap_struct_features(hm2)
        else:
            struct1 = self.zero_struct.copy()
            struct2 = self.zero_struct.copy()

        img1 = torch.from_numpy(img1).unsqueeze(0).repeat(3, 1, 1)
        img2 = torch.from_numpy(img2).unsqueeze(0).repeat(3, 1, 1)
        hm1  = torch.from_numpy(hm1).unsqueeze(0)
        hm2  = torch.from_numpy(hm2).unsqueeze(0)
        struct1 = torch.from_numpy(struct1.astype(np.float32))
        struct2 = torch.from_numpy(struct2.astype(np.float32))

        return img1, img2, hm1, hm2, struct1, struct2, torch.tensor(label, dtype=torch.long)

# ======================================================================
# 4. Sample collection
# ======================================================================
def list_sample_dirs(class_dir):
    out = []
    for n in os.listdir(class_dir):
        p = os.path.join(class_dir, n)
        if os.path.isdir(p) and n != "label":
            out.append(p)

    label_dir = os.path.join(class_dir, "label")
    if os.path.isdir(label_dir):
        for n in os.listdir(label_dir):
            p = os.path.join(label_dir, n)
            if os.path.isdir(p):
                out.append(p)
    return out


def try_get_heatmaps(cls, name):
    d = os.path.join(PRIORS_ROOT, cls, name)
    h1 = os.path.join(d, "1_heat.png")
    h2 = os.path.join(d, "2_heat.png")
    return (
        h1 if os.path.exists(h1) else None,
        h2 if os.path.exists(h2) else None
    )


def collect(cls, label):
    cls_dir = os.path.join(DATA_ROOT, cls)
    out = []

    for sd in list_sample_dirs(cls_dir):
        p1, p2 = os.path.join(sd, "1.png"), os.path.join(sd, "2.png")

        if not (os.path.exists(p1) and os.path.exists(p2)):
            continue

        name = os.path.basename(sd)
        h1, h2 = try_get_heatmaps(cls, name)

        out.append((p1, p2, h1, h2, label))

    return out
