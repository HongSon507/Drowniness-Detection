import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def to_rgb(x): return x.repeat(3, 1, 1)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  
    torch.backends.cudnn.deterministic = True  
    torch.cuda.empty_cache()
PROCESSED_DIR = "processed_data"
OUTPUT_DIR    = "results"
BATCH_SIZE    = 128
EPOCHS        = 50
LR            = 1e-3
PATIENCE      = 7       # early stopping
NUM_WORKERS   = 2      # number of workers
DEVICE        = torch.device("cuda")

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

torch.manual_seed(42)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Dataset ──────────────────────────────────────────────────────
class EyeDataset(Dataset):
    def __init__(self, split):
        self.images = np.load(f"{PROCESSED_DIR}/{split}_images.npy", mmap_mode="r")
        self.labels = np.load(f"{PROCESSED_DIR}/{split}_labels.npy")
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(to_rgb),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i]
        return self.tf(img), int(self.labels[i])


def get_loader(split, shuffle=False):
    return DataLoader(
        EyeDataset(split), batch_size=BATCH_SIZE, shuffle=shuffle,
        num_workers=NUM_WORKERS, persistent_workers=True, prefetch_factor=4,
    )


# ── Model ────────────────────────────────────────────────────────
def build_model():
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in m.features.parameters():
        p.requires_grad = False
    m.classifier = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(1280, 256), nn.ReLU(),
        nn.Dropout(0.2), nn.Linear(256, 2),
    )
    return m.to(DEVICE)


# ── Train / Eval ─────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None):
    model.train() if optimizer else model.eval()
    loss_sum, correct, n = 0, 0, 0
    all_p, all_y, all_prob = [], [], []

    ctx = torch.enable_grad() if optimizer else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out  = model(x)
            loss = criterion(out, y)

            if optimizer:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            loss_sum += loss.item() * len(y)
            prob = torch.softmax(out, 1)[:, 1]
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            n       += len(y)
            all_p.extend(pred.detach().cpu()); all_y.extend(y.detach().cpu()); all_prob.extend(prob.detach().cpu())

    y_true, y_pred, y_prob = map(np.array, (all_y, all_p, all_prob))
    return {
        "loss": loss_sum / n,
        "acc":  accuracy_score(y_true, y_pred),
        "f1":   f1_score(y_true, y_pred, zero_division=0),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec":  recall_score(y_true, y_pred, zero_division=0),
        "auc":  roc_auc_score(y_true, y_prob),
        "_y": y_true, "_p": y_pred, "_prob": y_prob,
    }


# ── Main ─────────────────────────────────────────────────────────
def main():
    train_loader = get_loader("train", shuffle=True)
    val_loader   = get_loader("val")
    test_loader  = get_loader("test")

    model     = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_auc, no_improve = 0, 0
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_auc": []}

    print(f"{'Epoch':>6} {'TrLoss':>8} {'TrAcc':>7} {'VaLoss':>8} {'VaAcc':>7} {'VaAUC':>7}")
    print("─" * 50)

    for epoch in range(1, EPOCHS + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer)
        va = run_epoch(model, val_loader, criterion)
        scheduler.step()

        for k, v in [("train_loss", tr["loss"]), ("train_acc", tr["acc"]),
                     ("val_loss",   va["loss"]), ("val_acc",   va["acc"]), ("val_auc", va["auc"])]:
            hist[k].append(v)

        print(f"{epoch:>6} {tr['loss']:>8.4f} {tr['acc']:>7.4f} {va['loss']:>8.4f} {va['acc']:>7.4f} {va['auc']:>7.4f}")

        if va["auc"] > best_auc + 1e-4:
            best_auc = va["auc"]
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pth")
            print(f"       ✔ saved (auc={best_auc:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}"); break

    # ── Test ─────────────────────────────────────────────────────
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model.pth", map_location=DEVICE))
    te = run_epoch(model, test_loader, criterion)

    print("\n── Test Results ──────────────────────────────────")
    print(classification_report(te["_y"], te["_p"], target_names=["awake", "sleepy"]))
    print(f"AUC: {te['auc']:.4f}")

    # ── Plots ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    e = range(1, len(hist["train_loss"]) + 1)

    axes[0].plot(e, hist["train_loss"], label="Train"); axes[0].plot(e, hist["val_loss"], label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(e, hist["train_acc"], label="Train"); axes[1].plot(e, hist["val_acc"], label="Val")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(e, hist["val_auc"], color="green"); axes[2].set_title("Val AUC"); axes[2].grid(alpha=0.3)

    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/curves.png", dpi=100); plt.close()

    fpr, tpr, _ = roc_curve(te["_y"], te["_prob"])
    plt.figure(figsize=(5, 4)); plt.plot(fpr, tpr, lw=2, label=f"AUC={te['auc']:.4f}")
    plt.plot([0,1],[0,1],"k--"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC – Test"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/roc.png", dpi=100); plt.close()

    cm = confusion_matrix(te["_y"], te["_p"])
    fig, ax = plt.subplots(figsize=(4, 3)); im = ax.imshow(cm, cmap="Blues"); plt.colorbar(im)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["awake","sleepy"]); ax.set_yticklabels(["awake","sleepy"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i,j], ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/confusion.png", dpi=100); plt.close()

    print(f"\nSaved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()