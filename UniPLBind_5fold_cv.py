"""
Protein-level outer 5-fold cross-validation for UniPLBind.

Protocol implemented here
---------------------------
1. Only the development dataset (e.g., ATP-387 or PPI-352) is used.
2. KFold creates five outer protein-level folds.
3. In each outer fold, the held-out proteins are used only for outer-fold evaluation.
4. The remaining outer-training proteins are split into primary-training (80%)
   and meta-validation (20%) subsets for the V-Net meta-learning procedure.
5. Early stopping and checkpoint selection use only the meta-validation subset.
6. The independent ATP-41/PPI-70 benchmark is not read or used by this script.

Before running, check DATASET_CONFIGS and set all paths for the desired dataset.
This file preserves the current two-logit classifier architecture, but fixes the
outer-fold split, padded-residue masking, meta-validation checkpoint selection,
and differentiable V-Net virtual update.
"""

from __future__ import annotations

import copy
import csv
import datetime
import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split
from torch.func import functional_call
from torch.utils.data import DataLoader, Dataset

from Resnet1 import VNet


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_SEED = 2026
N_OUTER_FOLDS = 5
META_VALIDATION_RATIO = 0.20
BATCH_SIZE = 16
MAX_EPOCHS = 30
PATIENCE = 5
WARMUP_EPOCHS = 3
MODEL_LR = 1e-4
VNET_LR = 1e-4
INNER_LR = 1e-4
GAMMA = 2.0
MIN_WEIGHT = 0.1
NUM_WORKERS = 0

# Set one or both datasets here. ATP-387 is configured using the paths in the
# uploaded script. Complete the PPI-352 paths before adding it to ACTIVE_DATASETS.
ACTIVE_DATASETS = ["ATP-387"]

DATASET_CONFIGS: Dict[str, Dict[str, object]] = {
    "ATP-387": {
        "list_file": r"E:/fengzhen/NucGMTL-main/DataSet/ATP/387_41/ATP387.txt",
        "prost_dir": r"E:/fengzhen/embedding_ATP1/ProstT5_embedding_387_41",
        "ankh_dir": r"E:/fengzhen/embedding_ATP1/Ankh_embedding_387_41",
        "label_dir": r"E:/fengzhen/embedding_ATP1/label_387_41",
        "alpha": (1.0, 10.0),
    },
    "PPI-352": {
        # Replace the three embedding/label directories below with your PPI paths.
        "list_file": r"E:/fengzhen/NucGMTL-main/DataSet/PPI/Train352-Test70/PPI-Train352.txt",
        "prost_dir": r"SET_YOUR_PPI_PROSTT5_EMBEDDING_DIRECTORY",
        "ankh_dir": r"SET_YOUR_PPI_ANKH_EMBEDDING_DIRECTORY",
        "label_dir": r"SET_YOUR_PPI_LABEL_DIRECTORY",
        "alpha": (1.0, 5.0),
    },
}

OUTPUT_ROOT = Path("RGMTL") / "five_fold_cv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = torch.cuda.is_available()


# -----------------------------------------------------------------------------
# Reproducibility and input utilities
# -----------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    """Set all random seeds used by the outer-fold experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_config(config: Dict[str, object], dataset_name: str) -> None:
    """Fail early if a required data path was not configured."""
    required_keys = ("list_file", "prost_dir", "ankh_dir", "label_dir")
    for key in required_keys:
        value = str(config[key])
        if "SET_YOUR" in value:
            raise ValueError(
                f"Please set DATASET_CONFIGS['{dataset_name}']['{key}'] before running."
            )
        if not Path(value).exists():
            raise FileNotFoundError(
                f"Configured path does not exist for {dataset_name}: {key} = {value}"
            )


def read_protein_ids(list_file: str) -> List[str]:
    """Read protein identifiers from the existing two-line-per-protein list file."""
    with open(list_file, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    if len(lines) % 2 != 0:
        raise ValueError(
            f"Expected an even number of lines in {list_file}, but found {len(lines)}."
        )

    protein_ids: List[str] = []
    for index in range(0, len(lines), 2):
        fields = lines[index].split()
        if not fields:
            raise ValueError(f"Empty protein-name line at line {index + 1} in {list_file}.")
        protein_ids.append(fields[0].strip())

    if len(set(protein_ids)) != len(protein_ids):
        raise ValueError("Duplicate protein identifiers were found in the development list.")
    return sorted(protein_ids)


# -----------------------------------------------------------------------------
# Dataset and batching
# -----------------------------------------------------------------------------
def coll_padding(batch_data):
    """Pad sequences and return valid sequence lengths for masking."""
    batch_data.sort(key=lambda item: len(item[0]), reverse=True)

    features = [item[0] for item in batch_data]
    labels = [item[1] for item in batch_data]
    task_ids = [item[2] for item in batch_data]
    lengths = torch.tensor([len(item[0]) for item in batch_data], dtype=torch.long)

    features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    task_ids = torch.nn.utils.rnn.pad_sequence(task_ids, batch_first=True, padding_value=0)
    return features, labels, task_ids, lengths


class BioinformaticsDataset(Dataset):
    """Loads concatenated ProstT5 and Ankh residue embeddings for protein IDs."""

    def __init__(self, protein_ids: Sequence[str], config: Dict[str, object]):
        self.protein_ids = list(protein_ids)
        self.prost_dir = Path(str(config["prost_dir"]))
        self.ankh_dir = Path(str(config["ankh_dir"]))
        self.label_dir = Path(str(config["label_dir"]))

    def __len__(self) -> int:
        return len(self.protein_ids)

    def __getitem__(self, index: int):
        protein_id = self.protein_ids[index]
        prost_path = self.prost_dir / f"{protein_id}.data"
        ankh_path = self.ankh_dir / f"{protein_id}.data"
        label_path = self.label_dir / f"{protein_id}.label"

        for path in (prost_path, ankh_path, label_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing input file: {path}")

        prost = torch.tensor(
            pd.read_csv(prost_path, header=None).values.astype(np.float32), dtype=torch.float32
        )
        ankh = torch.tensor(
            pd.read_csv(ankh_path, header=None).values.astype(np.float32), dtype=torch.float32
        )
        labels = torch.tensor(
            pd.read_csv(label_path, header=None).values.astype(np.int64).reshape(-1),
            dtype=torch.long,
        )

        valid_length = min(prost.size(0), ankh.size(0), labels.size(0))
        if valid_length == 0:
            raise ValueError(f"Protein {protein_id} has an empty embedding or label file.")

        feature = torch.cat((prost[:valid_length], ankh[:valid_length]), dim=1)
        labels = labels[:valid_length]
        task_ids = torch.zeros(valid_length, dtype=torch.long)
        return feature, labels, task_ids


def build_loader(
    protein_ids: Sequence[str],
    config: Dict[str, object],
    *,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        BioinformaticsDataset(protein_ids, config),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=coll_padding,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


# -----------------------------------------------------------------------------
# UniPLBind architecture (kept compatible with the uploaded script)
# -----------------------------------------------------------------------------
class AttentionModel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.q = nn.Linear(in_dim, out_dim)
        self.k = nn.Linear(in_dim, out_dim)
        self.v = nn.Linear(in_dim, out_dim)
        self.norm_factor = out_dim ** -0.5

    @staticmethod
    def create_src_lengths_mask(batch_size: int, src_lengths: torch.Tensor) -> torch.Tensor:
        max_length = int(src_lengths.max().item())
        indices = torch.arange(max_length, device=src_lengths.device).unsqueeze(0)
        indices = indices.expand(batch_size, max_length)
        lengths = src_lengths.unsqueeze(1).expand(batch_size, max_length)
        return indices < lengths

    def masked_softmax(self, scores: torch.Tensor, src_lengths: torch.Tensor) -> torch.Tensor:
        batch_size = scores.size(0)
        key_mask = self.create_src_lengths_mask(batch_size, src_lengths).to(scores.device)
        scores = scores.masked_fill(~key_mask.unsqueeze(1), float("-inf"))
        return F.softmax(scores.float(), dim=-1)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        query = self.q(features)
        key = self.k(features)
        value = self.v(features)
        scores = torch.bmm(query, key.transpose(1, 2)) * self.norm_factor
        attention = self.masked_softmax(scores, lengths)
        return torch.bmm(attention, value) + value


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.ms1cnn1 = nn.Conv1d(input_dim, 512, 1, padding="same")
        self.ms1cnn2 = nn.Conv1d(512, 256, 1, padding="same")
        self.ms1cnn3 = nn.Conv1d(256, 128, 1, padding="same")

        self.ms2cnn1 = nn.Conv1d(input_dim, 512, 3, padding="same")
        self.ms2cnn2 = nn.Conv1d(512, 256, 3, padding="same")
        self.ms2cnn3 = nn.Conv1d(256, 128, 3, padding="same")

        self.ms3cnn1 = nn.Conv1d(input_dim, 512, 5, padding="same")
        self.ms3cnn2 = nn.Conv1d(512, 256, 5, padding="same")
        self.ms3cnn3 = nn.Conv1d(256, 128, 5, padding="same")

        self.relu = nn.ReLU(inplace=True)
        self.attention1 = AttentionModel(512, 128)
        self.attention2 = AttentionModel(256, 128)
        self.attention3 = AttentionModel(128, 128)

    def forward(self, protein_input: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        shared_input = protein_input.permute(0, 2, 1)

        m1 = self.relu(self.ms1cnn1(shared_input))
        m2 = self.relu(self.ms2cnn1(shared_input))
        m3 = self.relu(self.ms3cnn1(shared_input))
        s1 = self.attention1((m1 + m2 + m3).permute(0, 2, 1), lengths)

        m1 = self.relu(self.ms1cnn2(m1))
        m2 = self.relu(self.ms2cnn2(m2))
        m3 = self.relu(self.ms3cnn2(m3))
        s2 = self.attention2((m1 + m2 + m3).permute(0, 2, 1), lengths)

        m1 = self.relu(self.ms1cnn3(m1))
        m2 = self.relu(self.ms2cnn3(m2))
        m3 = self.relu(self.ms3cnn3(m3))
        s3 = self.attention3((m1 + m2 + m3).permute(0, 2, 1), lengths)

        mscnn = (m1 + m2 + m3).permute(0, 2, 1)
        return mscnn + s1 + s2 + s3


class Module(nn.Module):
    def __init__(self, input_dim: int = 2560):
        super().__init__()
        self.feature_extractor = FeatureExtractor(input_dim)
        self.task_fc = nn.Sequential(
            nn.Linear(128, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 64),
            nn.Dropout(0.5),
            nn.Linear(64, 2),
        )

    def forward(self, protein_input: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(protein_input, lengths)
        return self.task_fc(features)


# -----------------------------------------------------------------------------
# Loss, masking, meta-learning, and evaluation
# -----------------------------------------------------------------------------
def sequence_mask(lengths: torch.Tensor, max_length: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(max_length, device=device).unsqueeze(0)
    return positions < lengths.to(device).unsqueeze(1)


def valid_logits_and_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
    lengths: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    valid_mask = sequence_mask(lengths, logits.size(1), logits.device)
    return logits[valid_mask], labels.to(logits.device)[valid_mask]


def focal_loss_per_residue(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    cross_entropy = F.cross_entropy(logits, labels, reduction="none")
    pt = torch.exp(-cross_entropy)
    alpha_t = alpha[labels.long()]
    return alpha_t * (1.0 - pt).pow(gamma) * cross_entropy


def call_vnet(vnet: nn.Module, detached_losses: torch.Tensor) -> torch.Tensor:
    """Generate one dynamic weight per valid residue."""
    vnet_input = detached_losses.reshape(-1, 1)
    dummy = torch.zeros_like(vnet_input)
    return vnet(vnet_input, dummy, dummy).reshape(-1)


def next_batch(iterator, loader: DataLoader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


@torch.no_grad()
def evaluate_meta_loss(
    model: nn.Module,
    loader: DataLoader,
    alpha: torch.Tensor,
) -> float:
    """Evaluate unweighted focal loss on the held-out meta-validation proteins."""
    model.eval()
    loss_sum = 0.0
    residue_count = 0

    for features, labels, _, lengths in loader:
        features = features.to(DEVICE, non_blocking=PIN_MEMORY)
        labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)
        logits = model(features, lengths)
        logits_valid, labels_valid = valid_logits_and_labels(logits, labels, lengths)
        per_residue_loss = focal_loss_per_residue(logits_valid, labels_valid, alpha, GAMMA)
        loss_sum += per_residue_loss.sum().item()
        residue_count += labels_valid.numel()

    if residue_count == 0:
        raise RuntimeError("No valid residues were found in the meta-validation loader.")
    return loss_sum / residue_count


def train_one_outer_fold(
    primary_train_files: Sequence[str],
    meta_validation_files: Sequence[str],
    config: Dict[str, object],
    checkpoint_path: Path,
) -> Tuple[nn.Module, nn.Module, float, int]:
    """Train one CV fold using primary/meta splits inside the outer training set."""
    model = Module().to(DEVICE)
    vnet = VNet(input=1, hidden1=100, output=1, num_classes=1).to(DEVICE)
    optimizer_model = torch.optim.Adam(model.parameters(), lr=MODEL_LR)
    optimizer_vnet = torch.optim.Adam(vnet.parameters(), lr=VNET_LR)
    alpha = torch.tensor(config["alpha"], dtype=torch.float32, device=DEVICE)

    train_loader = build_loader(primary_train_files, config, shuffle=True)
    meta_loader = build_loader(meta_validation_files, config, shuffle=True)
    if len(train_loader) == 0 or len(meta_loader) == 0:
        raise RuntimeError("Primary training or meta-validation loader is empty.")

    best_meta_loss = float("inf")
    best_epoch = -1
    no_improvement_epochs = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        vnet.train()
        train_loss_sum = 0.0
        train_residue_count = 0
        meta_iterator = iter(meta_loader)

        for features, labels, _, lengths in train_loader:
            features = features.to(DEVICE, non_blocking=PIN_MEMORY)
            labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)

            # Warm-up: focal-loss training without dynamic weights.
            if epoch < WARMUP_EPOCHS:
                optimizer_model.zero_grad(set_to_none=True)
                logits = model(features, lengths)
                logits_valid, labels_valid = valid_logits_and_labels(logits, labels, lengths)
                per_residue_loss = focal_loss_per_residue(logits_valid, labels_valid, alpha, GAMMA)
                final_loss = per_residue_loss.mean()
                final_loss.backward()
                optimizer_model.step()

            else:
                # 1) Differentiable virtual model update for V-Net meta-learning.
                logits = model(features, lengths)
                logits_valid, labels_valid = valid_logits_and_labels(logits, labels, lengths)
                per_residue_loss = focal_loss_per_residue(logits_valid, labels_valid, alpha, GAMMA)
                virtual_weights = torch.clamp(
                    call_vnet(vnet, per_residue_loss.detach()), min=MIN_WEIGHT, max=1.0
                )
                virtual_train_loss = (virtual_weights * per_residue_loss).mean()

                parameter_dict = OrderedDict(model.named_parameters())
                gradients = torch.autograd.grad(
                    virtual_train_loss,
                    tuple(parameter_dict.values()),
                    create_graph=True,
                    allow_unused=False,
                )
                fast_parameters = OrderedDict(
                    (name, parameter - INNER_LR * gradient)
                    for (name, parameter), gradient in zip(parameter_dict.items(), gradients)
                )

                # 2) Update V-Net from an independent meta-validation batch.
                (meta_features, meta_labels, _, meta_lengths), meta_iterator = next_batch(
                    meta_iterator, meta_loader
                )
                meta_features = meta_features.to(DEVICE, non_blocking=PIN_MEMORY)
                meta_labels = meta_labels.to(DEVICE, non_blocking=PIN_MEMORY)

                meta_logits = functional_call(
                    model,
                    fast_parameters,
                    args=(meta_features, meta_lengths),
                    strict=False,
                )
                meta_logits_valid, meta_labels_valid = valid_logits_and_labels(
                    meta_logits, meta_labels, meta_lengths
                )
                meta_per_residue_loss = focal_loss_per_residue(
                    meta_logits_valid, meta_labels_valid, alpha, GAMMA
                )
                meta_loss = meta_per_residue_loss.mean()

                optimizer_vnet.zero_grad(set_to_none=True)
                meta_loss.backward()
                optimizer_vnet.step()

                # 3) Recompute constant final weights with the updated V-Net,
                #    then update the primary prediction model.
                optimizer_model.zero_grad(set_to_none=True)
                logits = model(features, lengths)
                logits_valid, labels_valid = valid_logits_and_labels(logits, labels, lengths)
                per_residue_loss = focal_loss_per_residue(logits_valid, labels_valid, alpha, GAMMA)
                with torch.no_grad():
                    final_weights = torch.clamp(
                        call_vnet(vnet, per_residue_loss.detach()),
                        min=MIN_WEIGHT,
                        max=1.0,
                    )
                final_loss = (final_weights * per_residue_loss).mean()
                final_loss.backward()
                optimizer_model.step()

            train_loss_sum += final_loss.detach().item() * labels_valid.numel()
            train_residue_count += labels_valid.numel()

        meta_loss = evaluate_meta_loss(model, meta_loader, alpha)
        mean_train_loss = train_loss_sum / max(train_residue_count, 1)
        print(
            f"Epoch {epoch + 1:02d}/{MAX_EPOCHS} | "
            f"train loss: {mean_train_loss:.6f} | meta loss: {meta_loss:.6f}"
        )

        if meta_loss < best_meta_loss:
            best_meta_loss = meta_loss
            best_epoch = epoch + 1
            no_improvement_epochs = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "vnet_state_dict": copy.deepcopy(vnet.state_dict()),
                    "best_meta_loss": best_meta_loss,
                    "best_epoch": best_epoch,
                },
                checkpoint_path,
            )
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs >= PATIENCE:
                print(f"Early stopping after epoch {epoch + 1}.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    vnet.load_state_dict(checkpoint["vnet_state_dict"])
    return model, vnet, float(checkpoint["best_meta_loss"]), int(checkpoint["best_epoch"])


@torch.no_grad()
def evaluate_outer_fold(model: nn.Module, outer_test_files: Sequence[str], config: Dict[str, object]) -> Dict[str, float]:
    """Evaluate the selected model once on the held-out outer fold."""
    model.eval()
    loader = build_loader(outer_test_files, config, shuffle=False)

    all_labels: List[np.ndarray] = []
    all_probabilities: List[np.ndarray] = []

    for features, labels, _, lengths in loader:
        features = features.to(DEVICE, non_blocking=PIN_MEMORY)
        labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)
        logits = model(features, lengths)
        probabilities = F.softmax(logits, dim=-1)[..., 1]
        valid_mask = sequence_mask(lengths, logits.size(1), DEVICE)

        all_labels.append(labels[valid_mask].detach().cpu().numpy())
        all_probabilities.append(probabilities[valid_mask].detach().cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_probabilities)
    y_pred = (y_score >= 0.5).astype(np.int64)

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1_score = 2.0 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0.0

    precision_curve, recall_curve, _ = metrics.precision_recall_curve(y_true, y_score)
    return {
        "SEN": sensitivity * 100.0,
        "SPE": specificity * 100.0,
        "ACC": metrics.accuracy_score(y_true, y_pred) * 100.0,
        "PRE": precision * 100.0,
        "F1": f1_score * 100.0,
        "MCC": metrics.matthews_corrcoef(y_true, y_pred),
        "AUROC": metrics.roc_auc_score(y_true, y_score),
        "AUPRC": metrics.auc(recall_curve, precision_curve),
    }


# -----------------------------------------------------------------------------
# Outer 5-fold cross-validation orchestration
# -----------------------------------------------------------------------------
def save_cv_results(
    fold_records: List[Dict[str, float]],
    dataset_name: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_df = pd.DataFrame(fold_records)
    metric_columns = ["SEN", "SPE", "ACC", "PRE", "F1", "MCC", "AUROC", "AUPRC"]

    summary = {
        "Dataset": dataset_name,
        "Fold": "Mean ± SD",
        "N_outer_train": "",
        "N_primary_train": "",
        "N_meta_validation": "",
        "N_outer_test": "",
        "Best epoch": "",
        "Best meta loss": "",
    }
    for metric in metric_columns:
        summary[metric] = f"{fold_df[metric].mean():.3f} ± {fold_df[metric].std(ddof=1):.3f}"

    fold_csv = output_dir / f"{dataset_name}_five_fold_metrics.csv"
    summary_csv = output_dir / f"{dataset_name}_five_fold_summary.csv"
    fold_df.to_csv(fold_csv, index=False, float_format="%.6f")
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    print("\nFive-fold cross-validation summary")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"Fold-wise metrics saved to: {fold_csv}")
    print(f"Summary saved to: {summary_csv}")


def run_five_fold_cv(dataset_name: str, config: Dict[str, object]) -> None:
    validate_config(config, dataset_name)
    all_files = read_protein_ids(str(config["list_file"]))
    if len(all_files) < N_OUTER_FOLDS:
        raise ValueError(
            f"{dataset_name} contains only {len(all_files)} proteins; cannot use {N_OUTER_FOLDS}-fold CV."
        )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"{dataset_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    kfold = KFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=BASE_SEED)
    fold_records: List[Dict[str, float]] = []

    print(f"\n{'=' * 72}\nDataset: {dataset_name} | proteins: {len(all_files)} | device: {DEVICE}\n{'=' * 72}")

    for fold_index, (outer_train_indices, outer_test_indices) in enumerate(kfold.split(all_files), start=1):
        fold_seed = BASE_SEED + fold_index
        set_global_seed(fold_seed)

        outer_train_files = [all_files[index] for index in outer_train_indices]
        outer_test_files = [all_files[index] for index in outer_test_indices]
        primary_train_files, meta_validation_files = train_test_split(
            outer_train_files,
            test_size=META_VALIDATION_RATIO,
            random_state=fold_seed,
            shuffle=True,
        )

        print(
            f"\n--- Outer fold {fold_index}/{N_OUTER_FOLDS} ---\n"
            f"primary train: {len(primary_train_files)} | "
            f"meta validation: {len(meta_validation_files)} | "
            f"outer test: {len(outer_test_files)}"
        )

        checkpoint_path = output_dir / f"{dataset_name}_fold_{fold_index}_best.pt"
        model, _, best_meta_loss, best_epoch = train_one_outer_fold(
            primary_train_files,
            meta_validation_files,
            config,
            checkpoint_path,
        )
        fold_metrics = evaluate_outer_fold(model, outer_test_files, config)
        fold_metrics.update(
            {
                "Dataset": dataset_name,
                "Fold": fold_index,
                "N_outer_train": len(outer_train_files),
                "N_primary_train": len(primary_train_files),
                "N_meta_validation": len(meta_validation_files),
                "N_outer_test": len(outer_test_files),
                "Best epoch": best_epoch,
                "Best meta loss": best_meta_loss,
            }
        )
        fold_records.append(fold_metrics)

        metrics_text = ", ".join(f"{key}={value:.4f}" for key, value in fold_metrics.items() if key in {
            "SEN", "SPE", "ACC", "PRE", "F1", "MCC", "AUROC", "AUPRC"
        })
        print(f"Fold {fold_index} outer-test metrics: {metrics_text}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_cv_results(fold_records, dataset_name, output_dir)


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    print(f"Using device: {DEVICE}")

    for active_dataset in ACTIVE_DATASETS:
        if active_dataset not in DATASET_CONFIGS:
            raise KeyError(f"Unknown dataset name in ACTIVE_DATASETS: {active_dataset}")
        run_five_fold_cv(active_dataset, DATASET_CONFIGS[active_dataset])
