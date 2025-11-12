# JD Search Recommendation System

This repository contains datasets, data preparation scripts, and model implementations for the **JD Search** recommendation system project. The project focuses on analyzing user behavior, product metadata, and building recommendation models using sequence encoders and hybrid neural collaborative filtering (NCF).

---

## Datasets

### 1. JD Search - User Behaviour Data
- **Link:** [User Behaviour Data](https://drive.google.com/file/d/1xvfq2v_hhQ9G1FOczV6gl79R9HXpf_kd/view?usp=drive_link)  
- **Description:** Contains users’ historical behavior, search queries, candidate products, and test labels.

### 2. JD Search - Product Metadata
- **Link:** [Product Metadata](https://drive.google.com/file/d/1J2Xz8t2NuLuLuP-q5X76a0vvjwkON5ND/view?usp=drive_link)  
- **Description:** Contains detailed metadata about products including hierarchical categories, brand, and shop information.

---

## Data Preparation

**Colab Folder:**  [Data Preparation Scripts](https://drive.google.com/drive/folders/1AMDvPmo_xNiDujWt6sWogyMU4NXERlSU?usp=sharing)  

| File | Description |
|------|-------------|
| `Sampling.ipynb` | Prepares the raw product and user behaviour data by filtering and sampling relevant subsets. Reduces data size while preserving meaningful interactions. |
| `Data Preprocessing.ipynb` | Parses and cleans the sampled data. Transforms concatenated fields into structured lists, removes invalid entries, filters unmatched product IDs, and encodes interaction types numerically. Outputs parquet files for downstream modeling. |
| `Merged Feature Engineering.ipynb` | Generates features from user behavior and product metadata to support recommendation model training, including category diversity, purchase behavior, and category popularity. |
| `PCA and Clustering.ipynb` | Performs dimensionality reduction (PCA) and user segmentation (KMeans) on engineered features. Outputs include cluster assignments and visualizations of user groups. |

---

## Models

**Colab Folder:** [Model Implementations](https://drive.google.com/drive/folders/1t7LMqohnXh4kc1Ga45eLqUztqNFOSPaH?usp=sharing)  

| File | Description |
|------|-------------|
| `1.1 Sequence Encoder (LSTM + Attention).ipynb` | Implements a sequence encoder using LSTM with attention to model user interaction histories. Includes data loading, preprocessing, embeddings, and neural architecture. |
| `1.2 GRU Sequence Encoder.ipynb` | Implements a GRU-based sequence encoder with attention. Incorporates time gaps, type encoding, and sequence bucketing for user embeddings. |
| `2. Candidate Embedding.ipynb` | Encodes product metadata into integer indices for model training. Constructs lookup tensors for brand, shop, and category embeddings. |
| `3.1 Hybrid NCF with LSTM.ipynb` | Implements a hybrid recommendation model combining Neural Collaborative Filtering (NCF) with LSTM sequence encoding. Uses candidate features, user sequences, and engineered meta features for personalized ranking. Handles data preprocessing, model definition, training, and evaluation. |

---

## Usage

1. Download the datasets from the provided links.  
2. Run the data preparation notebooks in order:  
   - `Sampling.ipynb` → `Data Preprocessing.ipynb` → `Merged Feature Engineering.ipynb` → `PCA and Clustering.ipynb`  
3. Use the model notebooks for training and evaluation:  
   - Sequence encoders: `1.1 Sequence Encoder` / `1.2 GRU Sequence Encoder`  
   - Candidate embeddings: `2. Candidate Embedding.ipynb`  
   - Hybrid NCF model: `3.1 Hybrid NCF with LSTM.ipynb`  


