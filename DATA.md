# Data guide

The repository includes the compact sequence-and-label benchmark files needed
to identify proteins and residue-level ground truth. Each protein occupies two
lines: the first contains its identifier and amino-acid sequence; the second
contains the same identifier and a binary label string of equal length.

## Included benchmark files

| Ligand/task | Development set | Independent test set | Additional included splits |
|---|---:|---:|---|
| Protein | PPI-Train352 (352) | PPI-Test70 (70) | PPI-Train335, PPI-Test60 |
| DNA | DNA_Train (573) | DNA_Test (129) | — |
| RNA | RNA_Train (495) | RNA_Test (117) | — |
| Peptide | PepPI-Train640 (640) | PepPI-Test639 (639) | — |
| ATP | ATP387 (387) | ATP41 (41) | ATP542, ATP202, ATP226, ATP17 |
| Small molecule | SMB_Train (240) | SMB_Test (355) | — |

Numbers in parentheses are protein-chain records read from the distributed
files. The historical filename `ATP226.txt` contains 230 two-line records; the
file is retained unchanged for traceability.

## Embeddings are not included

ProstT5 and Ankh residue embeddings are generated files and are too large for
the source repository. Their expected layout and matrix format are documented
in [`embeddings/README.md`](embeddings/README.md). Generate them with
`scripts/extract_embeddings.py`, or point a local configuration to an existing
compatible embedding directory.

## Data provenance and reuse

The benchmark files are redistributed to support method reproduction. They may
originate from third-party benchmark collections; users remain responsible for
following the terms and citation requirements of the original data sources.
The included labels are for computational research and do not constitute
clinical annotations or recommendations.

