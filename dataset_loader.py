"""
dataset_loader.py
-----------------
Loads and aligns cloud images with ERA5 tabular data.

Alignment Rule (confirmed by user):
  - Cloudburst   : use ALL 492 images  → first 492 tabular rows (Label=1)
  - Non-Cloudburst: first 550 images   → ALL  550 tabular rows (Label=0)
  - Total paired dataset : 492 + 550 = 1042 samples
"""

import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


# ──────────────────────────────────────────────────────────────
#  Constants  (edit these paths if you move files)
# ──────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
CB_IMAGE_DIR     = os.path.join(BASE_DIR, "dataset", "cloudburst")
NON_CB_IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "non_cloudburst")
EXCEL_PATH       = os.path.join(BASE_DIR, "ERA5_Single_Sheet main.xlsx")

# Tabular feature columns (drop Id and Label)
FEATURE_COLS = ["Latitude", "Longitude", "Temperature", "Precipitation"]

IMG_SIZE = 224   # resize all images to 224×224


# ──────────────────────────────────────────────────────────────
#  Image Transforms
# ──────────────────────────────────────────────────────────────
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet stats
                         std =[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ──────────────────────────────────────────────────────────────
#  Helper: load & clean Excel
# ──────────────────────────────────────────────────────────────
def load_tabular_data():
    """
    Reads the Excel file, strips the merged header row,
    renames columns, and returns two DataFrames:
        cb_tab   → 500 cloudburst rows   (Label=1)
        ncb_tab  → 550 non-cloudburst rows (Label=0)
    """
    raw = pd.read_excel(EXCEL_PATH, header=None)

    # Row 0 is the merged title; Row 1 is the real column header
    raw.columns = raw.iloc[1]           # use row-index 1 as header
    raw = raw.iloc[2:].reset_index(drop=True)   # drop title + header rows

    raw.columns = ["Id", "Latitude", "Longitude", "Temperature", "Precipitation", "Label"]
    raw = raw.astype({"Latitude": float, "Longitude": float,
                      "Temperature": float, "Precipitation": float,
                      "Label": int})

    cb_tab  = raw[raw["Label"] == 1].reset_index(drop=True)   # 500 rows
    ncb_tab = raw[raw["Label"] == 0].reset_index(drop=True)   # 550 rows

    print(f"[Tabular]  Cloudburst rows  : {len(cb_tab)}")
    print(f"[Tabular]  Non-Cloudburst rows: {len(ncb_tab)}")
    return cb_tab, ncb_tab


# ──────────────────────────────────────────────────────────────
#  Helper: collect & sort image paths by class
# ──────────────────────────────────────────────────────────────
def collect_image_paths():
    def sorted_pngs(folder):
        files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
        files.sort()   # deterministic alphabetical order
        return [os.path.join(folder, f) for f in files]

    cb_imgs  = sorted_pngs(CB_IMAGE_DIR)
    ncb_imgs = sorted_pngs(NON_CB_IMAGE_DIR)

    print(f"[Images]   Cloudburst images   : {len(cb_imgs)}")
    print(f"[Images]   Non-Cloudburst images: {len(ncb_imgs)}")
    return cb_imgs, ncb_imgs


# ──────────────────────────────────────────────────────────────
#  Alignment  (class-level index pairing)
# ──────────────────────────────────────────────────────────────
def build_aligned_pairs():
    """
    Returns a list of tuples:
        (image_path: str, tabular_row: pd.Series, label: int)

    Alignment:
        Cloudburst     → n_cb  = min(492 imgs, 500 tab) = 492
        Non-Cloudburst → n_ncb = min(579 imgs, 550 tab) = 550
    """
    cb_tab, ncb_tab = load_tabular_data()
    cb_imgs, ncb_imgs = collect_image_paths()

    # --- Cloudburst pairs ---
    n_cb = min(len(cb_imgs), len(cb_tab))
    cb_pairs = [
        (cb_imgs[i], cb_tab.iloc[i], 1)
        for i in range(n_cb)
    ]

    # --- Non-Cloudburst pairs ---
    n_ncb = min(len(ncb_imgs), len(ncb_tab))
    ncb_pairs = [
        (ncb_imgs[i], ncb_tab.iloc[i], 0)
        for i in range(n_ncb)
    ]

    all_pairs = cb_pairs + ncb_pairs
    print(f"\n[Alignment] Cloudburst pairs    : {len(cb_pairs)}")
    print(f"[Alignment] Non-Cloudburst pairs: {len(ncb_pairs)}")
    print(f"[Alignment] Total paired samples: {len(all_pairs)}\n")
    return all_pairs


# ──────────────────────────────────────────────────────────────
#  Tabular Normalizer (fit on train, apply to val/test)
# ──────────────────────────────────────────────────────────────
class TabularNormalizer:
    def __init__(self):
        self.mean = None
        self.std  = None

    def fit(self, rows):
        """rows: list of pd.Series"""
        vals = np.array([[r[c] for c in FEATURE_COLS] for r in rows], dtype=np.float32)
        self.mean = vals.mean(axis=0)
        self.std  = vals.std(axis=0) + 1e-8

    def transform(self, row):
        """row: pd.Series → normalized np.ndarray (4,)"""
        vals = np.array([row[c] for c in FEATURE_COLS], dtype=np.float32)
        return (vals - self.mean) / self.std


# ──────────────────────────────────────────────────────────────
#  PyTorch Dataset
# ──────────────────────────────────────────────────────────────
class CloudburstDataset(Dataset):
    def __init__(self, pairs, normalizer, img_transform):
        """
        pairs        : list of (img_path, tab_row, label)
        normalizer   : fitted TabularNormalizer
        img_transform: torchvision transforms
        """
        self.pairs         = pairs
        self.normalizer    = normalizer
        self.img_transform = img_transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, tab_row, label = self.pairs[idx]

        # --- Image ---
        image = Image.open(img_path).convert("RGB")
        image = self.img_transform(image)

        # --- Tabular ---
        tab = torch.tensor(
            self.normalizer.transform(tab_row),
            dtype=torch.float32
        )

        # --- Label ---
        label = torch.tensor(label, dtype=torch.float32)

        return image, tab, label


# ──────────────────────────────────────────────────────────────
#  Factory: build train / val / test DataLoaders
# ──────────────────────────────────────────────────────────────
def get_dataloaders(batch_size=32, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Splits aligned pairs into train / val / test sets,
    fits the tabular normalizer on train only,
    and returns three DataLoaders.
    """
    all_pairs = build_aligned_pairs()
    n = len(all_pairs)

    # Reproducible shuffle
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    all_pairs = [all_pairs[i] for i in idx]

    # Split counts
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val

    train_pairs = all_pairs[:n_train]
    val_pairs   = all_pairs[n_train: n_train + n_val]
    test_pairs  = all_pairs[n_train + n_val:]

    print(f"[Split] Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")

    # Fit normalizer on TRAIN only
    normalizer = TabularNormalizer()
    normalizer.fit([p[1] for p in train_pairs])

    # Build datasets
    train_ds = CloudburstDataset(train_pairs, normalizer, TRAIN_TRANSFORMS)
    val_ds   = CloudburstDataset(val_pairs,   normalizer, EVAL_TRANSFORMS)
    test_ds  = CloudburstDataset(test_pairs,  normalizer, EVAL_TRANSFORMS)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    return train_loader, val_loader, test_loader, normalizer


# ──────────────────────────────────────────────────────────────
#  Quick audit (run directly to verify data)
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pairs = build_aligned_pairs()
    print("=== First 3 paired samples ===")
    for i in range(3):
        img_path, tab_row, label = pairs[i]
        print(f"  [{i}] Label={label}  Image: {os.path.basename(img_path)}")
        print(f"       Tab: Lat={tab_row['Latitude']:.2f}, Lon={tab_row['Longitude']:.2f}, "
              f"Temp={tab_row['Temperature']:.2f}, Precip={tab_row['Precipitation']:.6f}")

    print("\n=== Last 3 paired samples ===")
    for i in range(-3, 0):
        img_path, tab_row, label = pairs[i]
        print(f"  [{i}] Label={label}  Image: {os.path.basename(img_path)}")
        print(f"       Tab: Lat={tab_row['Latitude']:.2f}, Lon={tab_row['Longitude']:.2f}, "
              f"Temp={tab_row['Temperature']:.2f}, Precip={tab_row['Precipitation']:.6f}")
