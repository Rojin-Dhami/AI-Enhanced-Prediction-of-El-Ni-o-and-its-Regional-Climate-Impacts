"""
Training script for the CNN-TCN model.

Assumes you already have, from Section 5.7 (Feature Engineering and Tensor
Construction), a saved array of input windows and their two targets:

    X.npy         shape (N, 12, 30, 100, 4)   float32   -- input tensors
    y_enso.npy    shape (N,)                  float32   -- Nino 3.4 @ 6mo lead
    y_impact.npy  shape (N, 2, 340)           float32   -- [precip, temp] anomalies
    dates.npy     shape (N,)                  datetime64[M] or similar, the
                                               window END month, used only to
                                               build the chronological split

If you haven't saved these yet, the Section-5.7 pipeline should end by
writing exactly these four arrays (e.g. via np.save or a single .npz).

Run:
    python train.py --data_dir /path/to/tensors --epochs 100
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import CNNTCN, multitask_loss


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ENSODataset(Dataset):
    def __init__(self, X, y_enso, y_impact):
        self.X = torch.from_numpy(X).float()
        self.y_enso = torch.from_numpy(y_enso).float()
        self.y_impact = torch.from_numpy(y_impact).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_enso[idx], self.y_impact[idx]


def chronological_split(dates: np.ndarray):
    """
    Matches Section 4.4.4 exactly:
        Train:      1980-2018
        Validation: 2019-2022
        Test:       2023-2025
    `dates` should be the window-end month for each sample.
    """
    years = dates.astype("datetime64[Y]").astype(int) + 1970  # -> int year
    train_idx = np.where(years <= 2018)[0]
    val_idx = np.where((years >= 2019) & (years <= 2022))[0]
    test_idx = np.where((years >= 2023) & (years <= 2025))[0]
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def run_epoch(model, loader, optimizer, device, alpha, train: bool):
    model.train() if train else model.eval()
    total_loss = total_enso = total_impact = 0.0
    n_batches = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for X, y_enso, y_impact in loader:
            X, y_enso, y_impact = X.to(device), y_enso.to(device), y_impact.to(device)

            if train:
                optimizer.zero_grad()

            pred_enso, pred_impact = model(X)
            loss, l_enso, l_impact = multitask_loss(
                pred_enso, y_enso, pred_impact, y_impact, alpha=alpha
            )

            if train:
                loss.backward()
                # TCNs can produce sharp gradients through the dilated stack;
                # clip to keep training stable, especially early on.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item()
            total_enso += l_enso.item()
            total_impact += l_impact.item()
            n_batches += 1

    return (total_loss / n_batches,
            total_enso / n_batches,
            total_impact / n_batches)


def anomaly_correlation(pred: np.ndarray, true: np.ndarray) -> float:
    """ACC as defined in Eq. 4-6 / 5-3 -- Pearson correlation."""
    pred, true = pred.flatten(), true.flatten()
    return float(np.corrcoef(pred, true)[0, 1])


def evaluate_test_set(model, loader, device):
    model.eval()
    all_pred_enso, all_true_enso = [], []
    with torch.no_grad():
        for X, y_enso, y_impact in loader:
            X = X.to(device)
            pred_enso, _ = model(X)
            all_pred_enso.append(pred_enso.cpu().numpy())
            all_true_enso.append(y_enso.numpy())
    pred = np.concatenate(all_pred_enso)
    true = np.concatenate(all_true_enso)
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    acc = anomaly_correlation(pred, true)
    return rmse, acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Directory containing X.npy, y_enso.npy, "
                              "y_impact.npy, dates.npy")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.5,
                         help="Weight on ENSO loss vs impact loss (Eq. 4-19). "
                              "Tune this on the validation set per Section 4.7.")
    parser.add_argument("--patience", type=int, default=15,
                         help="Early stopping patience on validation loss.")
    parser.add_argument("--out", type=str, default="best_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load pre-built tensors (Section 5.7 output) ---
    X = np.load(f"{args.data_dir}/X.npy")
    y_enso = np.load(f"{args.data_dir}/y_enso.npy")
    y_impact = np.load(f"{args.data_dir}/y_impact.npy")
    dates = np.load(f"{args.data_dir}/dates.npy")

    train_idx, val_idx, test_idx = chronological_split(dates)
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

    train_ds = ENSODataset(X[train_idx], y_enso[train_idx], y_impact[train_idx])
    val_ds = ENSODataset(X[val_idx], y_enso[val_idx], y_impact[val_idx])
    test_ds = ENSODataset(X[test_idx], y_enso[test_idx], y_impact[test_idx])

    # small dataset (~450 train samples) -> small batch size, no need to shuffle test/val
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = CNNTCN(
        in_channels=4,
        spatial_dim=64,
        tcn_channels=(64, 64, 64, 64),
        tcn_kernel_size=3,
        num_impact_cells=y_impact.shape[-1],
        dropout=0.3,          # start aggressive -- your dataset is tiny (N~450)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_enso, train_impact = run_epoch(
            model, train_loader, optimizer, device, args.alpha, train=True
        )
        val_loss, val_enso, val_impact = run_epoch(
            model, val_loader, optimizer, device, args.alpha, train=False
        )
        scheduler.step(val_loss)

        print(f"Epoch {epoch:3d} | "
              f"train_loss={train_loss:.4f} (enso={train_enso:.4f}, impact={train_impact:.4f}) | "
              f"val_loss={val_loss:.4f} (enso={val_enso:.4f}, impact={val_impact:.4f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.out)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no val improvement for {args.patience} epochs).")
                break

    # --- Final test-set evaluation with best checkpoint ---
    model.load_state_dict(torch.load(args.out))
    test_rmse, test_acc = evaluate_test_set(model, test_loader, device)
    print(f"\nTest RMSE (Nino 3.4): {test_rmse:.4f}")
    print(f"Test ACC  (Nino 3.4): {test_acc:.4f}")
    print("\nCompare against your Method 1/2 XGBoost baselines:")
    print("  Method 1 (Regularized): Test RMSE=0.9077, ACC=0.8163")
    print("  Method 2 (Expanded):    Test RMSE=0.5299, ACC=0.9207")


if __name__ == "__main__":
    main()
