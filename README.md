# UniPLBind

**Unified protein–ligand binding-site prediction with collaborative protein
language-model representations and meta-learned dynamic residue reweighting**

UniPLBind is a sequence-based framework for residue-level binding-site
prediction across six ligand modalities: **protein, DNA, RNA, peptide, ATP,
and small molecule**. It integrates complementary ProstT5 and Ankh residue
representations, decodes local and long-range sequence context with a
multi-scale CNN–attention network, and uses a V-Net meta-learning branch to
adapt residue weights under severe class imbalance.

## Method overview

1. **Collaborative PLM representation** — 1,024-dimensional ProstT5 and
   1,536-dimensional Ankh residue embeddings are concatenated into a
   2,560-dimensional input representation.
2. **Multi-scale sequence decoding** — parallel 1D convolutions with kernel
   sizes 1, 3, and 5 capture residue patterns at different local scales.
3. **Context aggregation** — masked self-attention complements the convolutional
   features with long-range dependencies while excluding padded positions.
4. **Dynamic residue reweighting** — a V-Net learns loss-dependent residue
   weights from an inner meta-validation split. The held-out outer fold is not
   used for weighting, early stopping, or checkpoint selection.
5. **Unified task protocol** — the same architecture and protein-level
   evaluation workflow can be applied to all six binding-site tasks.

## Repository layout

```text
UniPLBind/
├── configs/                 # Example paths and task-specific settings
├── data/                    # Compact sequence and residue-label benchmarks
├── embeddings/              # Format documentation; large arrays are omitted
├── scripts/
│   ├── extract_embeddings.py
│   └── run_cv.py            # Nested protein-level cross-validation
├── src/uniplbind/
│   ├── data.py
│   ├── meta.py
│   └── model.py
├── tests/
├── DATA.md
└── requirements*.txt
```

Legacy exploratory scripts, IDE metadata, temporary results, checkpoints,
figures, and duplicate data copies from the original working directory are not
part of this curated release.

## Installation

Python 3.10 or later is recommended.

```bash
git clone https://github.com/fengzhen01/UniPLBind.git
cd UniPLBind
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Feature extraction additionally requires the pretrained-model dependencies:

```bash
pip install -r requirements-extraction.txt
```

Install the PyTorch build appropriate for the local CPU/CUDA environment when
the default PyPI build is not suitable.

## Data and embeddings

The compact benchmark sequence/label files are included. See [DATA.md](DATA.md)
for the task-to-file mapping and dataset notes.

Precomputed PLM embeddings are **not uploaded** because they are large generated
artifacts. For example, the two embeddings for the DNA development set can be
generated as follows:

```bash
python scripts/extract_embeddings.py --model prostt5 \
  --input data/dna/DNA_Train.txt \
  --output embeddings/dna/prostt5 \
  --labels embeddings/dna/labels

python scripts/extract_embeddings.py --model ankh \
  --input data/dna/DNA_Train.txt \
  --output embeddings/dna/ankh \
  --labels embeddings/dna/labels
```

Run the same commands for the corresponding independent test file when final
independent evaluation is required. Model weights are downloaded by the
respective upstream libraries unless a local ProstT5 path is passed with
`--prostt5-model`.

## Protein-level cross-validation

Copy the example configuration if local paths need to be changed:

```bash
cp configs/datasets.example.json configs/datasets.local.json
```

Then run a task, for example DNA:

```bash
python scripts/run_cv.py \
  --config configs/datasets.local.json \
  --task dna \
  --seed 2026 \
  --folds 5
```

On Windows PowerShell, use the same command on one line or replace the shell
continuation character as appropriate. Available task keys are `protein`,
`dna`, `rna`, `peptide`, `atp`, and `small_molecule`.

The runner implements:

- protein-level outer folds;
- an 80:20 primary-training/meta-validation split inside each outer fold;
- focal loss during warm-up;
- differentiable V-Net updates after warm-up;
- padded-residue masking;
- early stopping based only on meta-validation loss; and
- fold-wise plus mean/standard-deviation output for SEN, SPE, PRE, F1, MCC,
  AUROC, and AUPRC.

The independent test set listed in the configuration is deliberately not read
by `run_cv.py`; it remains sealed during model development.

## Smoke test

```bash
pip install pytest
pytest -q
```

## Reproducibility notes

- Generated embeddings must use the same PLM family and residue-token alignment
  described in `embeddings/README.md`.
- Task-specific focal-loss class weights are explicit in the JSON configuration.
- Random seeds, split counts, training budgets, and learning rates are command-line
  parameters rather than hard-coded machine paths.
- Results and checkpoints are written under `outputs/`, which is ignored by Git.

## Citation

The UniPLBind manuscript is in preparation. If this repository supports your
work, please cite the accompanying article once its bibliographic information
is available. The README will be updated with the final citation and DOI.

## Scope

UniPLBind is intended for computational research. Predicted binding residues
require experimental validation and should not be treated as clinical or
therapeutic evidence.

