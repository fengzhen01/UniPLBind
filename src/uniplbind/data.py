"""Dataset parsing and batching utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def read_benchmark(path: str | Path) -> list[tuple[str, str, str]]:
    """Read the two-lines-per-protein benchmark format."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) % 2:
        raise ValueError(f"Expected an even number of lines in {path}.")
    records: list[tuple[str, str, str]] = []
    for offset in range(0, len(lines), 2):
        name, sequence = lines[offset].split()[:2]
        label_name, labels = lines[offset + 1].split()[:2]
        if name != label_name:
            raise ValueError(f"Identifier mismatch near line {offset + 1}: {name} != {label_name}")
        if len(sequence) != len(labels):
            raise ValueError(f"Sequence/label length mismatch for {name}.")
        records.append((name, sequence, labels))
    return records


class ResidueEmbeddingDataset(Dataset):
    def __init__(
        self,
        protein_ids: Sequence[str],
        prostt5_dir: str | Path,
        ankh_dir: str | Path,
        label_dir: str | Path,
    ) -> None:
        self.protein_ids = list(protein_ids)
        self.prostt5_dir = Path(prostt5_dir)
        self.ankh_dir = Path(ankh_dir)
        self.label_dir = Path(label_dir)

    def __len__(self) -> int:
        return len(self.protein_ids)

    def __getitem__(self, index: int):
        protein_id = self.protein_ids[index]
        paths = (
            self.prostt5_dir / f"{protein_id}.data",
            self.ankh_dir / f"{protein_id}.data",
            self.label_dir / f"{protein_id}.label",
        )
        if missing := [str(path) for path in paths if not path.exists()]:
            raise FileNotFoundError("Missing input file(s): " + ", ".join(missing))
        prostt5 = pd.read_csv(paths[0], header=None).to_numpy(dtype=np.float32)
        ankh = pd.read_csv(paths[1], header=None).to_numpy(dtype=np.float32)
        labels = pd.read_csv(paths[2], header=None).to_numpy(dtype=np.int64).reshape(-1)
        length = min(len(prostt5), len(ankh), len(labels))
        if length == 0:
            raise ValueError(f"Empty feature or label array for {protein_id}.")
        features = np.concatenate((prostt5[:length], ankh[:length]), axis=1)
        return torch.from_numpy(features), torch.from_numpy(labels[:length])


def collate_residues(batch):
    batch = sorted(batch, key=lambda item: item[0].size(0), reverse=True)
    features, labels = zip(*batch)
    lengths = torch.tensor([item.size(0) for item in features], dtype=torch.long)
    return (
        torch.nn.utils.rnn.pad_sequence(features, batch_first=True),
        torch.nn.utils.rnn.pad_sequence(labels, batch_first=True),
        lengths,
    )


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_residues,
        pin_memory=torch.cuda.is_available(),
    )

