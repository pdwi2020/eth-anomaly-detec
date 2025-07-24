# eth-anomaly-detec
# Unsupervised Anomaly Detection in Ethereum Transactions

This repository contains the source code and experimental results for the research paper, "Unsupervised Ethereum Anomaly Detection: A Comparative Study of Aggregated Features and Graph Neural Network Embeddings." The project investigates and contrasts two distinct feature representation strategies for identifying anomalous addresses in a large-scale Ethereum transaction dataset.

## Table of Contents
- [Project Overview](#project-overview)
- [Key Findings](#key-findings)
- [Methodology](#methodology)
  - [Feature Engineering 1: Aggregated Features](#feature-engineering-1-aggregated-features)
  - [Feature Engineering 2: GNN Embeddings](#feature-engineering-2-gnn-embeddings)
  - [Anomaly Detection Algorithms](#anomaly-detection-algorithms)
- [Repository Structure](#repository-structure)
- [Setup and Installation](#setup-and-installation)
- [Running the Experiments](#running-the-experiments)
- [Results](#results)
  - [Anomaly Score Distributions](#anomaly-score-distributions)
  - [Top-K Anomaly Comparison](#top-k-anomaly-comparison)
- [Citation](#citation)
- [License](#license)

## Project Overview

The integrity of blockchain platforms like Ethereum is threatened by illicit activities such as fraud, scams, and money laundering. Due to the scarcity of labeled data, unsupervised anomaly detection is a critical tool for monitoring network health. This project explores a fundamental question in this domain: **What is the most effective way to represent user (address) behavior for anomaly detection?**

We implement a complete pipeline to compare two approaches:
1.  **Aggregated Features:** Traditional hand-engineered statistical features that summarize the transactional history of an address (e.g., transaction count, average value, standard deviation).
2.  **GNN Embeddings:** Learned feature representations (embeddings) from a Graph Neural Network (GNN) that considers not only an address's own activity but also its position and relationships within the entire transaction graph.

The study applies three standard anomaly detection algorithms (Isolation Forest, One-Class SVM, LOF) to both feature sets and analyzes the significant differences in the results.

## Key Findings

Our primary finding is a **strong divergence** between the anomalies identified by the two feature sets.
- **Low Overlap:** The top 50 anomalies flagged by models using aggregated features have minimal overlap (0-18%) with those flagged by models using GNN embeddings.
- **Different Anomaly Types:** Qualitative analysis suggests that aggregated features are effective at catching outliers based on **magnitude** (e.g., addresses with extremely high/low transaction counts or values), while GNN embeddings are effective at catching **structural** outliers (e.g., addresses with unusual connection patterns, including potentially benign but central smart contracts).
- **No Single Best Approach:** The results indicate that neither feature representation is strictly superior; they offer complementary perspectives for a comprehensive anomaly detection strategy.

## Methodology

The project follows a systematic, multi-stage pipeline:

![Workflow Diagram](assets/workflow_diagram.png)  
*Optional: Create a simple workflow diagram and place it in an `assets` folder.*

### Feature Engineering 1: Aggregated Features
For each unique address, we compute 7 statistical features from its outgoing transaction history: `count`, `sum_value`, `mean_value`, `median_value`, `min_value`, `max_value`, and `std_value`.

### Feature Engineering 2: GNN Embeddings
1.  **Graph Construction:** A directed graph is built from the transaction data, with addresses as nodes and transactions as edges ($\approx$200k nodes, $\approx$2.9M edges).
2.  **GNN Encoder:** A 2-layer Graph Convolutional Network (GCN) is used as an encoder to learn node representations.
3.  **Unsupervised Training:** The GCN is trained on an unsupervised **link prediction** task for 1000 epochs. The model learns to predict whether a transaction (edge) is likely to exist between two addresses based on their features and position in the graph.
4.  **Embedding Extraction:** The final 64-dimensional embeddings are extracted from the trained GCN for every node in the graph.

### Anomaly Detection Algorithms
The following scikit-learn algorithms are trained on both feature sets after hyperparameter tuning with Optuna:
- `IsolationForest`
- `OneClassSVM`
- `LocalOutlierFactor`

## Repository Structure
```
.
├── main.ipynb                  # Main Jupyter notebook containing all code and experiments.
├── references.bib              # BibTeX file for paper citations.
├── paper/                      # Directory for the LaTeX paper draft.
│   └── main.tex
├── assets/                     # Directory for images and figures.
│   ├── figure_scores_comparison_2x3.png
│   ├── figure_contamination_sensitivity.png
│   └── ... (other figures)
├── requirements.txt            # Python dependencies.
└── README.md                   # This file.
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/pdwi2020/Unsupervised-Ethereum-Anomaly-Detection.git
    cd Unsupervised-Ethereum-Anomaly-Detection
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    This project requires PyTorch with CUDA support. Please follow the official instructions at [pytorch.org](https://pytorch.org/) to install PyTorch first, ensuring you select the correct CUDA version for your hardware.
    
    Then, install PyTorch Geometric and other packages:
    ```bash
    # Install PyTorch Geometric dependencies
    pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-$(python -c 'import torch; print(torch.__version__)').html

    # Install remaining packages
    pip install -r requirements.txt
    ```
    A sample `requirements.txt` would include:
    ```
    numpy
    pandas
    scikit-learn
    torch-geometric
    optuna
    matplotlib
    seaborn
    jupyterlab
    ```

4.  **Download Data:**
    Place the `eth_std_transactions.csv` dataset in the root directory of the project. Due to its size, the dataset is not included in this repository.

## Running the Experiments

All experiments are contained within the `main.ipynb` Jupyter notebook.
1.  Start Jupyter Lab:
    ```bash
    jupyter lab
    ```
2.  Open `main.ipynb` and execute the cells sequentially from top to bottom. The notebook is structured to:
    - Load and preprocess the data.
    - Build the transaction graph.
    - Train and evaluate anomaly detectors on aggregated features.
    - Train the GNN encoder and extract embeddings.
    - Retrain and evaluate anomaly detectors on GNN embeddings.
    - Generate all tables and figures used in the paper.

## Results

### Anomaly Score Distributions

The GNN embeddings enabled a much stronger separation of outliers for the LOF algorithm, as seen by the wide distribution of negative scores.

![Score Distributions](assets/figure_scores_comparison_2x3.png)
*Fig. 1: Comparison of test set anomaly score distributions for Aggregated vs. GNN features.*

### Top-K Anomaly Comparison

The study found minimal overlap (0-18%) between the top 50 anomalies identified by the two feature sets, highlighting that they capture fundamentally different types of anomalous behavior.

| Algorithm          | Overlap (Top 50) |
| ------------------ | ---------------- |
| Isolation Forest   | 18%              |
| One-Class SVM      | 6%               |
| Local Outlier Factor | 0%               |


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
