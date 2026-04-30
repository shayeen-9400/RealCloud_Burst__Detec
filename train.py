"""
train.py
--------
Training pipeline for the CNN+FNN cloudburst detection model.

Usage:
    python train.py

Outputs:
    - cloudburst_model.pth         (best model weights)
    - training_curves.png          (loss & accuracy plots)
    - training_log.csv             (epoch-by-epoch metrics)
"""

import os
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for Windows)
import matplotlib.pyplot as plt

from dataset_loader import get_dataloaders
from model import CloudburstFusionModel


# ──────────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────────
BATCH_SIZE      = 32
LEARNING_RATE   = 1e-3
EPOCHS          = 50
PATIENCE        = 10      # early stopping patience
MODEL_SAVE_PATH = "cloudburst_model.pth"
LOG_PATH        = "training_log.csv"
PLOT_PATH       = "training_curves.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using: {DEVICE}")


# ──────────────────────────────────────────────────────────────
#  Utility: compute accuracy from logits
# ──────────────────────────────────────────────────────────────
def binary_accuracy(logits, labels):
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds == labels).float().mean().item()


# ──────────────────────────────────────────────────────────────
#  One epoch: train
# ──────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for images, tabular, labels in loader:
        images  = images.to(DEVICE)
        tabular = tabular.to(DEVICE)
        labels  = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images, tabular)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc  += binary_accuracy(logits.detach(), labels)

    n = len(loader)
    return total_loss / n, total_acc / n


# ──────────────────────────────────────────────────────────────
#  One epoch: evaluate (val or test)
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    all_preds, all_labels = [], []

    for images, tabular, labels in loader:
        images  = images.to(DEVICE)
        tabular = tabular.to(DEVICE)
        labels  = labels.to(DEVICE)

        logits = model(images, tabular)
        loss   = criterion(logits, labels)

        total_loss += loss.item()
        total_acc  += binary_accuracy(logits, labels)
        all_preds.append(torch.sigmoid(logits).cpu())
        all_labels.append(labels.cpu())

    n = len(loader)
    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return total_loss / n, total_acc / n, all_preds, all_labels


# ──────────────────────────────────────────────────────────────
#  Plot training curves
# ──────────────────────────────────────────────────────────────
def plot_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cloudburst Model - Training Curves", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#4C72B0", lw=2)
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="#DD8452", lw=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train Acc", color="#4C72B0", lw=2)
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc",   color="#DD8452", lw=2)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"[Plot] Saved to {PLOT_PATH}")


# ──────────────────────────────────────────────────────────────
#  Final test evaluation with metrics
# ──────────────────────────────────────────────────────────────
def compute_metrics(preds, labels, threshold=0.5):
    preds_bin = (preds >= threshold).float()
    labels    = labels.float()

    tp = ((preds_bin == 1) & (labels == 1)).sum().item()
    tn = ((preds_bin == 0) & (labels == 0)).sum().item()
    fp = ((preds_bin == 1) & (labels == 0)).sum().item()
    fn = ((preds_bin == 0) & (labels == 1)).sum().item()

    accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return {"Accuracy": accuracy, "Precision": precision,
            "Recall": recall, "F1": f1}


# ──────────────────────────────────────────────────────────────
#  Main Training Loop
# ──────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("   Cloudburst Detection  -  CNN + FNN Fusion Model")
    print("=" * 60)

    # 1. Data
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        batch_size=BATCH_SIZE
    )

    # 2. Model
    model = CloudburstFusionModel().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total trainable parameters: {total_params:,}\n")

    # 3. Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 4. Logging
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss   = float("inf")
    patience_counter = 0

    csv_file = open(LOG_PATH, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Epoch", "Train Loss", "Val Loss", "Train Acc", "Val Acc", "LR"])

    # 5. Training loop
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'Train Acc':>10} | {'Val Acc':>10}")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss,   val_acc, _, _ = evaluate(model, val_loader, criterion)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        # Record
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        csv_writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}",
                              f"{train_acc:.4f}", f"{val_acc:.4f}", current_lr])
        csv_file.flush()

        print(f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>10.4f} | "
              f"{train_acc:>10.4f} | {val_acc:>10.4f}  [{elapsed:.1f}s]")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
            print(f"          [SAVED] Best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n[Early Stop] No improvement for {PATIENCE} epochs. Stopping.")
                break

    csv_file.close()
    print(f"\n[Log] Training log saved to {LOG_PATH}")

    # 6. Plot curves
    plot_curves(history)

    # 7. Test set evaluation
    print("\n" + "=" * 60)
    print("   Final Evaluation on Test Set")
    print("=" * 60)

    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion)
    metrics = compute_metrics(test_preds, test_labels)

    print(f"  Test Loss      : {test_loss:.4f}")
    print(f"  Test Accuracy  : {metrics['Accuracy']:.4f}  ({metrics['Accuracy']*100:.2f}%)")
    print(f"  Precision      : {metrics['Precision']:.4f}")
    print(f"  Recall         : {metrics['Recall']:.4f}")
    print(f"  F1 Score       : {metrics['F1']:.4f}")
    print("=" * 60)
    print(f"\n[Done] Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
