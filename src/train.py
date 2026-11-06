"""Hand-written training loop: forward -> loss -> backward -> step.

Usage:
    python -m src.train                          # full training with configs/default.yaml
    python -m src.train --resume                 # continue from checkpoints/last.pt
    python -m src.train --overfit-batch          # sanity check: can it memorize one batch?
    python -m src.train --set train.batch_size=4 data.image_size=384
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.dataset import BubbleDataset, collate, load_records
from src.losses import BCEDiceLoss
from src.model import UNet, count_parameters

LOG_FIELDS = ["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "lr", "seconds"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # Also seeds every CUDA device.


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(config: dict) -> UNet:
    return UNet(
        in_channels=config["model"]["in_channels"],
        base_channels=config["model"]["base_channels"],
    )


def build_loader(dataset, batch_size: int, num_workers: int, train: bool, seed: int = 0) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        # A last batch of one image would make BatchNorm statistics meaningless.
        drop_last=train and len(dataset) > batch_size,
    )


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp) -> float:
    model.train()  # BatchNorm uses batch statistics and updates its running averages.
    total_loss, total_images = 0.0, 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # Gradients accumulate unless cleared.
        with torch.autocast(device.type, enabled=use_amp):
            logits = model(images)  # Forward
            loss = criterion(logits, masks)  # Loss

        # The scaler multiplies the loss so small float16 gradients don't round to
        # zero, then un-scales them before the update. It does nothing when disabled.
        scaler.scale(loss).backward()  # Backward
        scaler.step(optimizer)  # Step
        scaler.update()

        total_loss += loss.item() * images.size(0)
        total_images += images.size(0)
    return total_loss / max(total_images, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp, threshold: float = 0.5) -> dict:
    model.eval()  # BatchNorm switches to the running averages learned in training.
    total_loss, dice_sum, iou_sum, count = 0.0, 0.0, 0.0, 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
        total_loss += loss.item() * images.size(0)

        preds = (torch.sigmoid(logits) > threshold).float().flatten(1)
        targets = masks.flatten(1)
        intersection = (preds * targets).sum(dim=1)
        pred_sum, target_sum = preds.sum(dim=1), targets.sum(dim=1)
        union = pred_sum + target_sum - intersection
        # A correctly empty prediction on a page with no bubbles counts as perfect.
        dice_sum += torch.where(
            pred_sum + target_sum > 0,
            2 * intersection / (pred_sum + target_sum).clamp(min=1),
            torch.ones_like(intersection),
        ).sum().item()
        iou_sum += torch.where(
            union > 0, intersection / union.clamp(min=1), torch.ones_like(intersection)
        ).sum().item()
        count += images.size(0)
    return {
        "loss": total_loss / max(count, 1),
        "dice": dice_sum / max(count, 1),
        "iou": iou_sum / max(count, 1),
    }


def cosine_schedule(optimizer, total_epochs: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine decay from the base LR to 0 over total_epochs.

    Written as a closed-form function of the epoch rather than CosineAnnealingLR,
    because that class restores its old cycle length from a checkpoint. Resuming
    with more epochs then drove the LR to 0 and back up again. Here the curve is
    rebuilt from the current config on every run.
    """
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: 0.5 * (1 + math.cos(math.pi * min(epoch, total_epochs) / total_epochs))
    )


def atomic_save(payload: dict, path: Path) -> None:
    """Write then rename, so a Kaggle session dying mid-save can't corrupt a checkpoint."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def overfit_one_batch(config: dict, device: torch.device, steps: int) -> list[float]:
    """Sanity check. A healthy model and loss must be able to memorize a single batch.

    If the loss doesn't approach zero here, suspect a bug in the code before
    blaming the data or the hyperparameters. Augmentation is off on purpose so the
    model sees identical inputs at every step.
    """
    records = [r for r in load_records(config["data"]["splits_dir"], "train") if r["boxes"]]
    batch_size = config["train"]["batch_size"]
    dataset = BubbleDataset(records[:batch_size], config["data"]["image_size"])
    batch = next(iter(build_loader(dataset, batch_size, 0, train=False)))
    images, masks = batch["image"].to(device), batch["mask"].to(device)

    model = build_model(config).to(device)
    criterion = BCEDiceLoss(config["loss"]["bce_weight"], config["loss"]["dice_weight"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["lr"])

    model.train()
    losses = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), masks)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step == 1 or step % 10 == 0 or step == steps:
            print(f"step {step:4d}/{steps}  loss {loss.item():.4f}", flush=True)
    print(f"Loss went from {losses[0]:.4f} to {losses[-1]:.4f}")
    return losses


def fit(config: dict, device: torch.device, resume: bool = False) -> dict:
    set_seed(config["seed"])
    data_cfg, train_cfg = config["data"], config["train"]

    train_set = BubbleDataset(
        load_records(data_cfg["splits_dir"], "train"), data_cfg["image_size"], config["augment"]
    )
    val_set = BubbleDataset(load_records(data_cfg["splits_dir"], "val"), data_cfg["image_size"])
    train_loader = build_loader(
        train_set, train_cfg["batch_size"], data_cfg["num_workers"], train=True, seed=config["seed"]
    )
    val_loader = build_loader(val_set, train_cfg["batch_size"], data_cfg["num_workers"], train=False)

    model = build_model(config).to(device)
    criterion = BCEDiceLoss(config["loss"]["bce_weight"], config["loss"]["dice_weight"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    scheduler = cosine_schedule(optimizer, train_cfg["epochs"])
    use_amp = bool(train_cfg["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(train_cfg["log_csv"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch, best_dice, stale_epochs = 1, -1.0, 0
    if resume:
        state = torch.load(checkpoint_dir / "last.pt", map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        # The optimizer restores the LR it last used, which came from the old curve.
        # Recompute it from the current one, or a longer resumed run starts at LR 0.
        for group, base_lr, lr_lambda in zip(
            optimizer.param_groups, scheduler.base_lrs, scheduler.lr_lambdas
        ):
            group["lr"] = base_lr * lr_lambda(scheduler.last_epoch)
        start_epoch = state["epoch"] + 1
        best_dice = state["best_dice"]
        stale_epochs = state["stale_epochs"]
        # Don't replay the same shuffles and augmentations as the epochs already done.
        set_seed(config["seed"] + start_epoch)
        train_loader.generator.manual_seed(config["seed"] + start_epoch)
        print(f"Resumed at epoch {start_epoch}, best val Dice so far {best_dice:.4f}")
    else:
        with open(log_path, "w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(LOG_FIELDS)

    print(
        f"Device {device} | AMP {use_amp} | {count_parameters(model):,} parameters | "
        f"{len(train_set)} train / {len(val_set)} val pages"
    )

    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        started = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        val = validate(model, val_loader, criterion, device, use_amp)
        scheduler.step()
        seconds = time.perf_counter() - started

        improved = val["dice"] > best_dice
        if improved:
            best_dice, stale_epochs = val["dice"], 0
        else:
            stale_epochs += 1

        with open(log_path, "a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(
                [epoch, f"{train_loss:.5f}", f"{val['loss']:.5f}", f"{val['dice']:.5f}",
                 f"{val['iou']:.5f}", f"{lr:.6f}", f"{seconds:.1f}"]
            )
        print(
            f"epoch {epoch:3d}/{train_cfg['epochs']}  train_loss {train_loss:.4f}  "
            f"val_loss {val['loss']:.4f}  val_dice {val['dice']:.4f}  val_iou {val['iou']:.4f}  "
            f"lr {lr:.2e}  {seconds:.0f}s{'  *best*' if improved else ''}",
            flush=True,
        )

        # best.pt holds only what inference needs. last.pt can resume training.
        if improved:
            atomic_save(
                {"model": model.state_dict(), "config": config, "epoch": epoch,
                 "val_dice": val["dice"], "val_iou": val["iou"]},
                checkpoint_dir / "best.pt",
            )
        atomic_save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
             "config": config, "epoch": epoch, "best_dice": best_dice,
             "stale_epochs": stale_epochs},
            checkpoint_dir / "last.pt",
        )

        if stale_epochs >= train_cfg["early_stopping_patience"]:
            print(f"Early stopping: no val Dice improvement for {stale_epochs} epochs.")
            break

    print(f"Done. Best val Dice {best_dice:.4f}. Checkpoint: {checkpoint_dir / 'best.pt'}")
    return {"best_dice": best_dice}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    # "extend" lets --set repeat: --set a=1 --set b=2 keeps both overrides.
    parser.add_argument(
        "--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE", dest="overrides"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overfit-batch", action="store_true")
    parser.add_argument("--overfit-steps", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = load_config(args.config, args.overrides)
    device = pick_device(args.device)
    if args.overfit_batch:
        set_seed(config["seed"])
        overfit_one_batch(config, device, args.overfit_steps)
    else:
        fit(config, device, resume=args.resume)


if __name__ == "__main__":
    main()
