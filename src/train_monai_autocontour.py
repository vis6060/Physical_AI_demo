from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from model import build_model, get_device


class SyntheticMRCTDataset(Dataset):
    def __init__(self, data_dir: str = "data/synthetic"):
        self.paths = sorted(Path(data_dir).glob("case_*.npz"))
        if not self.paths:
            raise FileNotFoundError(
                f"No .npz cases found in {data_dir}. Run generate_synthetic_mri_ct.py first."
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.paths[idx])
        x = np.stack([data["mri"], data["ct"]], axis=0).astype(np.float32)
        y = data["mask"].astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)


def dice_for_class(pred: torch.Tensor, target: torch.Tensor, cls: int) -> float:
    pred_cls = pred == cls
    target_cls = target == cls
    inter = (pred_cls & target_cls).sum().item()
    denom = pred_cls.sum().item() + target_cls.sum().item()
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    dices_organ, dices_tumor = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1)

        for b in range(pred.shape[0]):
            dices_organ.append(dice_for_class(pred[b], y[b], 1))
            dices_tumor.append(dice_for_class(pred[b], y[b], 2))

    return {
        "dice_organ": float(np.mean(dices_organ)),
        "dice_tumor": float(np.mean(dices_tumor)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/synthetic")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-model", type=str, default="outputs/model.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    dataset = SyntheticMRCTDataset(args.data_dir)

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))
            pbar.set_postfix(loss=float(np.mean(losses)))

        metrics = evaluate(model, val_loader, device)

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **metrics,
        }

        history.append(epoch_record)
        print(epoch_record)

    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": "monai_toy_autocontour",
            "model_version": "0.1.0",
            "input_channels": 2,
            "output_classes": 3,
        },
        args.out_model,
    )

    with open("outputs/training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Saved model to {args.out_model}")
    print("Saved training metrics to outputs/training_metrics.json")


if __name__ == "__main__":
    main()