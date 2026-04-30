"""
data_audit.py
-------------
Run this FIRST to verify data alignment before training.
Prints exact counts and shows sample paired entries.

Usage:
    python data_audit.py
"""

import os
import pandas as pd
import numpy as np

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
CB_IMAGE_DIR     = os.path.join(BASE_DIR, "dataset", "cloudburst")
NON_CB_IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "non_cloudburst")
EXCEL_PATH       = os.path.join(BASE_DIR, "ERA5_Single_Sheet main.xlsx")


def audit():
    sep = "=" * 65

    print(sep)
    print("  CLOUDBURST DATASET AUDIT")
    print(sep)

    # ── Image counts ──────────────────────────────────────────────
    cb_imgs  = sorted([f for f in os.listdir(CB_IMAGE_DIR)  if f.lower().endswith(".png")])
    ncb_imgs = sorted([f for f in os.listdir(NON_CB_IMAGE_DIR) if f.lower().endswith(".png")])

    print(f"\n[Images]")
    print(f"  dataset/cloudburst/     : {len(cb_imgs):>4} images  (Label=1)")
    print(f"  dataset/non_cloudburst/ : {len(ncb_imgs):>4} images  (Label=0)")

    # ── Tabular counts ───────────────────────────────────────────
    raw = pd.read_excel(EXCEL_PATH, header=None)
    raw.columns = raw.iloc[1]
    raw = raw.iloc[2:].reset_index(drop=True)
    raw.columns = ["Id", "Latitude", "Longitude", "Temperature", "Precipitation", "Label"]
    raw = raw.astype({"Latitude": float, "Longitude": float,
                      "Temperature": float, "Precipitation": float,
                      "Label": int})

    cb_tab  = raw[raw["Label"] == 1].reset_index(drop=True)
    ncb_tab = raw[raw["Label"] == 0].reset_index(drop=True)

    print(f"\n[Tabular] ERA5 Excel rows")
    print(f"  Cloudburst rows     (Label=1): {len(cb_tab):>4}")
    print(f"  Non-Cloudburst rows (Label=0): {len(ncb_tab):>4}")

    # ── Alignment summary ─────────────────────────────────────────
    n_cb  = min(len(cb_imgs),  len(cb_tab))    # 492
    n_ncb = min(len(ncb_imgs), len(ncb_tab))   # 550

    print(f"\n[Alignment]  (images = tabular, taking min per class)")
    print(f"  Cloudburst pairs    : min({len(cb_imgs)}, {len(cb_tab)}) = {n_cb}")
    print(f"  Non-Cloudburst pairs: min({len(ncb_imgs)}, {len(ncb_tab)}) = {n_ncb}")
    print(f"  TOTAL PAIRED DATASET: {n_cb + n_ncb}")

    # ── Split preview ─────────────────────────────────────────────
    total = n_cb + n_ncb
    print(f"\n[Train/Val/Test Split]  70% / 15% / 15%")
    print(f"  Train : ~{int(total*0.70)}")
    print(f"  Val   : ~{int(total*0.15)}")
    print(f"  Test  : ~{total - int(total*0.70) - int(total*0.15)}")

    # ── Sample paired entries ─────────────────────────────────────
    print(f"\n[Sample Pairs — Cloudburst (first 3)]")
    for i in range(min(3, n_cb)):
        print(f"  Image : {cb_imgs[i]}")
        row = cb_tab.iloc[i]
        print(f"  Tabular: Id={row['Id']}, Lat={row['Latitude']:.2f}, "
              f"Lon={row['Longitude']:.2f}, Temp={row['Temperature']:.2f}, "
              f"Precip={row['Precipitation']:.6f}, Label={int(row['Label'])}")
        print()

    print(f"[Sample Pairs — Non-Cloudburst (first 3)]")
    for i in range(min(3, n_ncb)):
        print(f"  Image : {ncb_imgs[i]}")
        row = ncb_tab.iloc[i]
        print(f"  Tabular: Id={row['Id']}, Lat={row['Latitude']:.2f}, "
              f"Lon={row['Longitude']:.2f}, Temp={row['Temperature']:.2f}, "
              f"Precip={row['Precipitation']:.6f}, Label={int(row['Label'])}")
        print()

    print(sep)
    print("  Audit complete. Run train.py to start training.")
    print(sep)


if __name__ == "__main__":
    audit()
