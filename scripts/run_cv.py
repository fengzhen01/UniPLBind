#!/usr/bin/env python
"""Protein-level nested five-fold evaluation for UniPLBind.

The outer test proteins are never used for optimization. Inside each outer
training fold, an independent meta-validation subset updates V-Net and selects
the checkpoint used for the outer-fold evaluation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split
from torch.func import functional_call

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from uniplbind.data import ResidueEmbeddingDataset, make_loader, read_benchmark  # noqa: E402
from uniplbind.meta import VNet  # noqa: E402
from uniplbind.model import UniPLBind  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Dataset JSON configuration.")
    parser.add_argument("--task", required=True, help="Task key in the configuration file.")
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "outputs")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--model-lr", type=float, default=1e-4)
    parser.add_argument("--vnet-lr", type=float, default=1e-4)
    parser.add_argument("--inner-lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--min-weight", type=float, default=0.1)
    return parser.parse_args()


def resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def load_task_config(config_path: Path, task: str) -> dict:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if task not in payload.get("tasks", {}):
        raise KeyError(f"Task '{task}' is absent from {config_path}.")
    config = dict(payload["tasks"][task])
    for key in ("development_file", "prostt5_dir", "ankh_dir", "label_dir"):
        config[key] = resolve_path(config[key], config_path)
        if not config[key].exists():
            raise FileNotFoundError(f"Missing {key}: {config[key]}")
    config.setdefault("input_dim", 2560)
    config.setdefault("class_alpha", [1.0, 5.0])
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sequence_mask(lengths: torch.Tensor, max_length: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(max_length, device=device).unsqueeze(0)
    return positions < lengths.to(device).unsqueeze(1)


def valid_residues(logits, labels, lengths):
    mask = sequence_mask(lengths, logits.size(1), logits.device)
    return logits[mask], labels.to(logits.device)[mask]


def focal_loss(logits, labels, alpha, gamma):
    cross_entropy = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    probability = torch.exp(-cross_entropy)
    return alpha[labels.long()] * (1.0 - probability).pow(gamma) * cross_entropy


def build_loader(ids, config, batch_size, shuffle):
    dataset = ResidueEmbeddingDataset(
        ids,
        config["prostt5_dir"],
        config["ankh_dir"],
        config["label_dir"],
    )
    return make_loader(dataset, batch_size=batch_size, shuffle=shuffle)


def cycle_next(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


@torch.no_grad()
def validation_loss(model, loader, alpha, gamma, device):
    model.eval()
    total = 0.0
    count = 0
    for features, labels, lengths in loader:
        features, labels = features.to(device), labels.to(device)
        logits, targets = valid_residues(model(features, lengths), labels, lengths)
        losses = focal_loss(logits, targets, alpha, gamma)
        total += losses.sum().item()
        count += targets.numel()
    return total / max(count, 1)


def train_fold(primary_ids, meta_ids, config, args, checkpoint, device):
    model = UniPLBind(input_dim=int(config["input_dim"])).to(device)
    vnet = VNet().to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr)
    vnet_optimizer = torch.optim.Adam(vnet.parameters(), lr=args.vnet_lr)
    alpha = torch.tensor(config["class_alpha"], dtype=torch.float32, device=device)
    train_loader = build_loader(primary_ids, config, args.batch_size, True)
    meta_loader = build_loader(meta_ids, config, args.batch_size, True)
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(args.epochs):
        model.train()
        vnet.train()
        meta_iterator = iter(meta_loader)
        for features, labels, lengths in train_loader:
            features, labels = features.to(device), labels.to(device)
            if epoch < args.warmup_epochs:
                model_optimizer.zero_grad(set_to_none=True)
                logits, targets = valid_residues(model(features, lengths), labels, lengths)
                loss = focal_loss(logits, targets, alpha, args.gamma).mean()
                loss.backward()
                model_optimizer.step()
                continue

            # Differentiable virtual step: V-Net weights the primary residues.
            logits, targets = valid_residues(model(features, lengths), labels, lengths)
            primary_losses = focal_loss(logits, targets, alpha, args.gamma)
            virtual_weights = vnet(primary_losses.detach()).clamp(args.min_weight, 1.0)
            virtual_loss = (virtual_weights * primary_losses).mean()
            parameters = OrderedDict(model.named_parameters())
            gradients = torch.autograd.grad(
                virtual_loss, tuple(parameters.values()), create_graph=True
            )
            fast_parameters = OrderedDict(
                (name, parameter - args.inner_lr * gradient)
                for (name, parameter), gradient in zip(parameters.items(), gradients)
            )

            # Update V-Net using an independent meta-validation batch.
            (meta_features, meta_labels, meta_lengths), meta_iterator = cycle_next(
                meta_iterator, meta_loader
            )
            meta_features, meta_labels = meta_features.to(device), meta_labels.to(device)
            meta_logits = functional_call(
                model, fast_parameters, args=(meta_features, meta_lengths), strict=False
            )
            meta_logits, meta_targets = valid_residues(meta_logits, meta_labels, meta_lengths)
            meta_loss = focal_loss(meta_logits, meta_targets, alpha, args.gamma).mean()
            vnet_optimizer.zero_grad(set_to_none=True)
            meta_loss.backward()
            vnet_optimizer.step()

            # Update the predictor with the newly estimated residue weights.
            model_optimizer.zero_grad(set_to_none=True)
            logits, targets = valid_residues(model(features, lengths), labels, lengths)
            primary_losses = focal_loss(logits, targets, alpha, args.gamma)
            with torch.no_grad():
                weights = vnet(primary_losses.detach()).clamp(args.min_weight, 1.0)
            loss = (weights * primary_losses).mean()
            loss.backward()
            model_optimizer.step()

        current_loss = validation_loss(model, meta_loader, alpha, args.gamma, device)
        print(f"epoch={epoch + 1:03d} meta_loss={current_loss:.6f}")
        if current_loss < best_loss:
            best_loss, best_epoch, stale_epochs = current_loss, epoch + 1, 0
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": copy.deepcopy(model.state_dict()),
                    "vnet": copy.deepcopy(vnet.state_dict()),
                    "epoch": best_epoch,
                    "meta_loss": best_loss,
                },
                checkpoint,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    vnet.load_state_dict(state["vnet"])
    return model, best_epoch, best_loss


@torch.no_grad()
def evaluate(model, ids, config, batch_size, device):
    model.eval()
    labels_all, scores_all = [], []
    for features, labels, lengths in build_loader(ids, config, batch_size, False):
        features, labels = features.to(device), labels.to(device)
        logits = model(features, lengths)
        mask = sequence_mask(lengths, logits.size(1), device)
        labels_all.append(labels[mask].cpu().numpy())
        scores_all.append(torch.softmax(logits, dim=-1)[..., 1][mask].cpu().numpy())
    y_true, y_score = np.concatenate(labels_all), np.concatenate(scores_all)
    y_pred = (y_score >= 0.5).astype(np.int64)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / max(tp + fp, 1)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision_curve, recall_curve, _ = metrics.precision_recall_curve(y_true, y_score)
    return {
        "SEN": sensitivity,
        "SPE": specificity,
        "PRE": precision,
        "F1": 2 * precision * sensitivity / max(precision + sensitivity, 1e-12),
        "MCC": metrics.matthews_corrcoef(y_true, y_pred),
        "AUROC": metrics.roc_auc_score(y_true, y_score),
        "AUPRC": metrics.auc(recall_curve, precision_curve),
    }


def write_results(records, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["fold", "primary_train", "meta_validation", "outer_test", "best_epoch", "meta_loss", "SEN", "SPE", "PRE", "F1", "MCC", "AUROC", "AUPRC"]
    with (output_dir / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    metric_names = ["SEN", "SPE", "PRE", "F1", "MCC", "AUROC", "AUPRC"]
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                name: {
                    "mean": float(np.mean([record[name] for record in records])),
                    "sd": float(np.std([record[name] for record in records], ddof=1)),
                }
                for name in metric_names
            },
            handle,
            indent=2,
        )


def main() -> None:
    args = parse_args()
    args.config = args.config.resolve()
    config = load_task_config(args.config, args.task)
    protein_ids = sorted(record[0] for record in read_benchmark(config["development_file"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output / args.task
    records = []
    splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    print(f"task={args.task} proteins={len(protein_ids)} device={device}")
    for fold, (train_indices, test_indices) in enumerate(splitter.split(protein_ids), start=1):
        fold_seed = args.seed + fold
        set_seed(fold_seed)
        outer_train = [protein_ids[index] for index in train_indices]
        outer_test = [protein_ids[index] for index in test_indices]
        primary, meta_validation = train_test_split(
            outer_train, test_size=0.20, random_state=fold_seed, shuffle=True
        )
        checkpoint = output_dir / "checkpoints" / f"fold_{fold}.pt"
        model, best_epoch, best_loss = train_fold(
            primary, meta_validation, config, args, checkpoint, device
        )
        record = evaluate(model, outer_test, config, args.batch_size, device)
        record.update(
            {
                "fold": fold,
                "primary_train": len(primary),
                "meta_validation": len(meta_validation),
                "outer_test": len(outer_test),
                "best_epoch": best_epoch,
                "meta_loss": best_loss,
            }
        )
        records.append(record)
        print(" ".join(f"{key}={value:.4f}" for key, value in record.items() if key in {"F1", "MCC", "AUROC", "AUPRC"}))
    write_results(records, output_dir)
    print(f"Results written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()

