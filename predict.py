"""
predict.py
----------
Run inference on a single new image + tabular row using the trained model.

Usage:
    python predict.py --image path/to/image.png --lat 29.0 --lon 84.0 --temp 287.5 --precip 0.006
"""

import argparse
import torch
import numpy as np
from PIL import Image

from dataset_loader import EVAL_TRANSFORMS, FEATURE_COLS
from model import CloudburstFusionModel

MODEL_PATH = "cloudburst_model.pth"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tabular normalization stats — update these with values printed during training
# (or load from a saved normalizer file if you add that step)
TAB_MEAN = np.array([31.0, 81.5, 288.0, 0.005], dtype=np.float32)
TAB_STD  = np.array([2.5,  3.5,  1.5,   0.003], dtype=np.float32)


def load_model():
    model = CloudburstFusionModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict(image_path, lat, lon, temp, precip, model):
    # --- Image ---
    image = Image.open(image_path).convert("RGB")
    image = EVAL_TRANSFORMS(image).unsqueeze(0).to(DEVICE)   # (1, 3, 224, 224)

    # --- Tabular (normalize using training stats) ---
    raw_tab = np.array([lat, lon, temp, precip], dtype=np.float32)
    norm_tab = (raw_tab - TAB_MEAN) / (TAB_STD + 1e-8)
    tabular = torch.tensor(norm_tab).unsqueeze(0).to(DEVICE)   # (1, 4)

    # --- Inference ---
    with torch.no_grad():
        logit = model(image, tabular)
        prob  = torch.sigmoid(logit).item()

    label  = "CLOUDBURST" if prob >= 0.5 else "NO CLOUDBURST"
    confidence = prob if prob >= 0.5 else 1 - prob

    return {"prediction": label, "probability": prob, "confidence": confidence}


def main():
    parser = argparse.ArgumentParser(description="Cloudburst Prediction")
    parser.add_argument("--image",  required=True,      help="Path to cloud image")
    parser.add_argument("--lat",    type=float, required=True, help="Latitude")
    parser.add_argument("--lon",    type=float, required=True, help="Longitude")
    parser.add_argument("--temp",   type=float, required=True, help="Temperature (K)")
    parser.add_argument("--precip", type=float, required=True, help="Precipitation")
    args = parser.parse_args()

    model = load_model()
    result = predict(args.image, args.lat, args.lon, args.temp, args.precip, model)

    print("\n" + "=" * 45)
    print(f"  Prediction  : {result['prediction']}")
    print(f"  Probability : {result['probability']:.4f}")
    print(f"  Confidence  : {result['confidence']*100:.2f}%")
    print("=" * 45)


if __name__ == "__main__":
    main()
