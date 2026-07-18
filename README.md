# UniPLBind: A Unified Model for Protein–Ligand Binding Site Prediction via Dynamic Residue Reweighting and Collaborative Optimization of PLM Representations

## 1. Overview

Accurate identification of protein–ligand binding residues is crucial for understanding molecular recognition, annotating protein functions, and facilitating drug discovery. Although numerous computational methods have been developed for ligand-specific binding-site prediction, most existing predictors are optimized for a single ligand type, which limits their applicability across heterogeneous ligand-binding scenarios.

To address this limitation, we propose **UniPLBind**, a unified sequence-based framework for residue-level protein–ligand binding-site prediction across diverse ligand modalities. UniPLBind integrates complementary protein language model (PLM) representations, dynamic residue reweighting, focal loss, and a multi-scale CNN–attention architecture to improve binding-site recognition under severe class imbalance.

Specifically, UniPLBind systematically evaluates multiple PLM embeddings and identifies complementary representation combinations for residue-level prediction. Dynamic residue reweighting adaptively adjusts the contribution of individual residues during training, helping the model focus on informative and hard-to-classify binding residues. Meanwhile, the multi-scale CNN module captures local contextual patterns, and the attention mechanism further models broader residue dependencies.

Extensive experiments across multiple ligand-binding tasks demonstrate that UniPLBind achieves strong performance among sequence-based methods and provides a unified solution for protein–ligand binding-site prediction without requiring explicit structural inputs.

# 1. Requirements
Python >= 3.10.6  

## 2. Datasets

UniPLBind is evaluated on multiple benchmark datasets covering diverse ligand-binding tasks, including protein, DNA, RNA, peptide, small-molecule, and ATP-binding residues. Each dataset contains protein sequences with residue-level binding annotations, where residues are labeled as binding or non-binding according to the corresponding ligand type.

The benchmark datasets used in this project include:

- **Protein-binding task**: protein–protein interaction binding-site prediction, evaluated on the PPI-70 benchmark.
- **DNA-binding task**: DNA-binding residue prediction, evaluated on the DPI-129 benchmark.
- **RNA-binding task**: RNA-binding residue prediction, evaluated on the RPI-117 benchmark.
- **Peptide-binding task**: peptide-binding residue prediction, evaluated on the PepPI-639 benchmark.
- **Small-molecule-binding task**: small-molecule-binding residue prediction, evaluated on the COACH355 benchmark.
- **ATP-binding task**: ATP-binding residue prediction, evaluated on ATP-related benchmark datasets, including ATP-17, ATP-41, and ATP-202.

These datasets provide heterogeneous ligand-binding scenarios and are used to evaluate the generalization ability of UniPLBind across different ligand modalities.

# 3. How to Use

## 3.1 Environment setup

UniPLBind is implemented in Python with PyTorch. Before running the model, please install the required dependencies and prepare the protein language model (PLM) environments used for feature extraction.

The PLM feature extraction scripts in this repository rely on pretrained protein language models. Please follow the official instructions of the corresponding PLM repositories to set up the required environments and download pretrained models.

For example:

- ProtTrans / ProtT5: https://github.com/agemagician/ProtTrans
- ESM / ESM2: https://github.com/facebookresearch/esm

After the required PLM environments are installed, the embedding extraction scripts can be used to generate residue-level PLM features for downstream training and prediction.

## 3.2 Feature extraction

To extract PLM embeddings, go to the `FeatureExtract` directory:

```bash
cd FeatureExtract

Run the following scripts according to the PLM features required by the experiment:python extract_prot1024.py


Run the following scripts according to the PLM features required by the experiment:

```bash
python extract_prot1024.py

This script extracts ProtTrans-based residue-level embeddings and stores the generated feature files in:

embedding/prot_embedding/

To extract ESM2-based residue-level embeddings, run:

python extractdata_esm1280.py

The generated ESM2 feature files will be stored in:

embedding/esm_embedding1280/

After feature extraction, return to the project root directory:

cd ..
3.3 Training and testing

To train and evaluate UniPLBind, run the main script from the project root directory:

python UniPLBind.py

This script performs model training and testing using the prepared datasets and extracted PLM embeddings.

For different experimental settings, please use the corresponding scripts provided in this repository. For example, five-fold cross-validation can be performed by running:

python UniPLBind_5fold_cv.py

Other scripts are provided for task-specific experiments, ablation studies, SHAP analysis, and t-SNE visualization. Users may modify the task configuration, dataset path, embedding path, and training parameters in the corresponding scripts according to their experimental requirements.

3.4 Output files

During training and testing, UniPLBind outputs residue-level prediction results, including predicted binding probabilities and binary prediction labels. Depending on the script used, the output files may include:

residue-level prediction scores;
evaluation metrics such as SEN, SPE, ACC, PRE, F1, MCC, AUROC, and AUPRC;
ROC and precision-recall curve data;
visualization results for residue-level score maps, heatmaps, SHAP attribution, and t-SNE analysis.

The output files are saved in the corresponding result or visualization folders specified in each script.

3.5 Notes
Please make sure that the extracted PLM embeddings match the protein sequences used in the corresponding datasets.
The sequence length should be consistent with the length of residue-level labels and prediction outputs.
GPU acceleration is recommended for PLM embedding extraction and model training.
Some scripts correspond to specific experiments reported in the manuscript, including ablation studies, SHAP analysis, and t-SNE visualization.
If you use different PLM embeddings or datasets, please update the corresponding file paths and input dimensions in the scripts.

If you have any questions or encounter problems when using UniPLBind, please open an issue in this repository.
   
