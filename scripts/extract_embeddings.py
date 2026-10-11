#!/usr/bin/env python
"""Extract residue-level ProstT5 or Ankh embeddings from a benchmark file."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from uniplbind.data import read_benchmark  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("prostt5", "ankh"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--prostt5-model",
        default="Rostlab/ProstT5",
        help="Hugging Face model identifier or local ProstT5 directory.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def save_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, delimiter=",", fmt="%.8g")


def write_labels(records, label_dir: Path) -> None:
    label_dir.mkdir(parents=True, exist_ok=True)
    for protein_id, _, labels in records:
        np.savetxt(
            label_dir / f"{protein_id}.label",
            np.fromiter((int(value) for value in labels), dtype=np.int64),
            fmt="%d",
        )


def extract_prostt5(records, output_dir: Path, model_name: str, device: str) -> None:
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name).to(device).eval()
    for index, (protein_id, sequence, _) in enumerate(records, start=1):
        destination = output_dir / f"{protein_id}.data"
        if destination.exists():
            continue
        spaced = " ".join(re.sub(r"[UZOB]", "X", sequence))
        encoded = tokenizer(spaced, add_special_tokens=True, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state[0]
        # ProstT5 adds one leading special token in the extraction protocol.
        embedding = hidden[1 : len(sequence) + 1].float().cpu().numpy()
        if len(embedding) != len(sequence):
            raise RuntimeError(f"Unexpected ProstT5 length for {protein_id}.")
        save_matrix(destination, embedding)
        print(f"prostt5 {index}/{len(records)} {protein_id}")


def extract_ankh(records, output_dir: Path, device: str) -> None:
    import ankh

    model, tokenizer = ankh.load_large_model()
    model = model.to(device).eval()
    for index, (protein_id, sequence, _) in enumerate(records, start=1):
        destination = output_dir / f"{protein_id}.data"
        if destination.exists():
            continue
        encoded = tokenizer.batch_encode_plus(
            [list(sequence)],
            add_special_tokens=True,
            padding=True,
            is_split_into_words=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state[0]
        embedding = hidden[: len(sequence)].float().cpu().numpy()
        if len(embedding) != len(sequence):
            raise RuntimeError(f"Unexpected Ankh length for {protein_id}.")
        save_matrix(destination, embedding)
        print(f"ankh {index}/{len(records)} {protein_id}")


def main() -> None:
    args = parse_args()
    records = read_benchmark(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    write_labels(records, args.labels)
    if args.model == "prostt5":
        extract_prostt5(records, args.output, args.prostt5_model, args.device)
    else:
        extract_ankh(records, args.output, args.device)


if __name__ == "__main__":
    main()

