## **Assessing Biases in Multimodal Recommender Systems: Analyzing the Accuracy-Fairness trade-off for Social Sustainability**

This repository contains the code and experimental setup for the paper *"Assessing Biases in Multimodal Recommender Systems: Analyzing the Accuracy-Fairness trade-off for Social Sustainability"*. The work investigates the interplay between multimodal features and demographic fairness in Recommender Systems (RS), evaluating both accuracy-oriented and fairness-oriented models on the MovieLens-1M dataset.

---

## Overview

Modern Multimodal Recommender Systems (MRS) enrich item representations with side-information such as product images, movie posters, or textual reviews, leading to improved recommendation accuracy. However, the impact of this multimodal enrichment on **demographic fairness** has remained largely unexplored.

This repository provides a unified experimental framework to:

- Assess how multimodal features affect accuracy, beyond-accuracy (diversity, coverage, popularity), and demographic fairness metrics.
- Investigate the accuracy–fairness trade-off under unimodal and multimodal conditions.
- Demonstrate that sophisticated fairness-aware architectures can leverage multimodal context to achieve simultaneous gains in both accuracy and fairness.

---

## Repository Structure

```
mmfairec/
├── confs/                   # YAML configuration files for each model
├── dataset/
│   └── mm_ml1m/             # MovieLens-1M dataset with multimodal features
├── recbole/                 # Extended RecBole source code (models, trainers, etc.)
├── run_bpr.py               # Grid-search runner for BPR
├── run_vbpr.py              # Grid-search runner for VBPR
├── run_lightgcn.py          # Grid-search runner for LightGCN
├── run_mmgcf.py             # Grid-search runner for MMGCF
├── run_focf.py              # Grid-search runner for FOCF
├── run_mmfocf.py            # Grid-search runner for MM-FOCF
├── run_nfcf.py              # Grid-search runner for NFCF
└── run_mmnfcf.py            # Grid-search runner for MM-NFCF
```

---

## Models

The repository evaluates six recommendation models, covering pure collaborative filtering, multimodal, and fairness-aware variants:

| Model | Type | Description |
|-------|------|-------------|
| **BPR** | CF baseline | Bayesian Personalized Ranking; learns user/item embeddings from interaction data only. |
| **VBPR** | Multimodal | Extends BPR via late-fusion of pre-trained visual and textual item features. |
| **LightGCN** | CF baseline | Learns user/item embeddings through GCN's message passing over layers. |
| **MMGCF** | Multimodal | Extends LightGCN via late-fusion of pre-trained visual and textual item features. |
| **FREEDOM** | Multimodal | Uses GCNs on freezed item-item modality-aware graph. |
| **FOCF** | Fairness-aware | Adds differentiable fairness objectives directly into the CF optimization loop. |
| **MM-FOCF** | Multimodal + fairness | FOCF extended with multimodal item embeddings via the late-fusion strategy described in the paper. |
| **NFCF** | Fairness-aware | Debiases user embeddings using a layer orthogonal to the sensitive attribute. |
| **MM-NFCF** | Multimodal + fairness | NFCF extended with multimodal item embeddings; achieves the best accuracy and fairness simultaneously. |


---

## Dataset

Experiments are conducted on **MovieLens-1M** (ML1M), extended with multimodal features.

| Property | Value |
|----------|-------|
| Users | 6,040 |
| Items | 3,196 |
| Interactions | 946,812 |
| Sparsity | 95.1% |
| Gender split (M / F) | 4,331 / 1,709 |
| Textual features | MiniLM (384-d) |
| Visual features | ViT (768-d) |

The multimodal features are sourced from the paper [See the Movie, Hear the Song, Read the Book](https://dl.acm.org/doi/10.1145/3705328.3748162) (RecSys '25). The sensitive attribute used for fairness evaluation is **gender**.

The dataset files are located under `dataset/mm_ml1m/` and follow the RecBole atomic file format.

---

## Installation

This project is built on top of [RecBole-FairRec](https://github.com/TangJiakai/RecBole-FairRec). Follow the steps below to set up your environment.

### Requirements

- Python >= 3.7.0
- PyTorch >= 1.11.0
- RecBole >= 1.0.1
- NumPy >= 1.20.3
- tqdm >= 4.62.3

### Step 1 – Clone RecBole-FairRec

```bash
git clone https://github.com/TangJiakai/RecBole-FairRec.git
cd RecBole-FairRec
```

### Step 2 – Install dependencies

```bash
pip install -r requirements.txt
```

Or install the core dependencies manually:

```bash
pip install recbole>=1.0.1 torch>=1.11.0 numpy>=1.20.3 tqdm>=4.62.3
```

### Step 3 – Clone this repository

```bash
cd ..
git clone https://github.com/swapUniba/mmfairec.git
cd mmfairec
```

> The `recbole/` folder inside this repository contains the extended RecBole source used for the experiments (including VBPR, MM-FOCF, and MM-NFCF implementations). It replaces the standard RecBole package for the purposes of these experiments.

---

## Usage

Each `run_*.py` script performs an automated **grid search** over the relevant hyperparameters, skipping configurations that have already been logged. Results are stored under a `log/<MODEL>/` directory.

### Running individual models

```bash
# Pure CF models
python run_bpr.py
python run_lightgcn.py

# Multimodal models
python run_vbpr.py
python run_mmgcf.py
python run_freedom.py

# Fairness-aware (CF only)
python run_focf.py
python run_nfcf.py

# Fairness-aware (multimodal)
python run_mmfocf.py
python run_mmnfcf.py
```

### Configuration files

Each model reads its base configuration from the corresponding YAML file in `confs/`. Hyperparameter values are appended programmatically by the runner script during the grid search. The following hyperparameters are tuned:

| Model | Hyperparameter(s) |
|-------|-------------------|
| BPR, VBPR | `weight_decay` ∈ {0.0001, 0.001, 0.01} |
| LightGCN, MMGCF | `weight_decay` ∈ {0.0001, 0.001, 0.01}; `n_layers` ∈ {1,2,3,4} |
| FREEDOM | `weight_decay` ∈ {0.0001, 0.001, 0.01}; `n_layers` ∈ {1,2,3,4}; `knn` ∈ {10,20}|
| FOCF, MM-FOCF | `weight_decay` ∈ {0.0001, 0.001, 0.01}; `fairness_weight` ∈ {0.0001, 0.001, 0.01}; `fairness_objective` ∈ {value, absolute, under, over, nonparity, none} |
| NFCF, MM-NFCF | `weight_decay` ∈ {0.0001, 0.001, 0.01}; `fairness_weight` ∈ {0.0001, 0.001, 0.01}; `dropout` ∈ {0, 0.2, 0.4, 0.6, 0.8} |

### Training protocol

All models share the following training settings:

- Train / Validation / Test split: **80 / 10 / 10**
- Optimizer: **Adam**, learning rate **0.001**
- Batch size: **2048**
- Embedding size: **64**
- Max epochs: **100** (early stopping after 10 epochs without NDCG@20 improvement)
- Validation metric: **NDCG@20**

---

## System Configuration

Experiments were run on:
- OS: Ubuntu 22.04
- CPU: Intel Xeon Silver 4309Y
- GPU: NVIDIA A16
- PyTorch 2.9, CUDA 12.8

---

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
