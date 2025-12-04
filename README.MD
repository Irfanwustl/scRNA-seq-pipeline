# scRNA-seq Pipeline

A clean, modular, and extensible scRNA-seq analysis pipeline written in Python.  
This package provides a **standard end-to-end workflow**—from QC to PCA, batch correction, clustering, and UMAP—built on top of Scanpy but designed for clarity, reproducibility, and extensibility.

---

## 🚀 Features

### 1. Preprocessing
- Gene/cell QC filtering  
- Mito/ribosomal/hemoglobin content filtering  
- Optional doublet removal using **Scrublet**  
- Highly variable gene (HVG) selection  
  - Default: **Seurat v3 HVGs on raw counts**  
- Normalization, log-transformation, scaling  
- PCA computation (`.obsm["X_pca"]`)

### 2. Batch Correction
- Harmony integration (default)  
- Optional: disable batch correction (`batch_method="none"`)

### 3. Clustering & Embedding
- Build KNN graph  
- Leiden or Louvain community detection  
- UMAP embedding (`.obsm["X_umap"]`)

### 4. Marker-based Annotation
- Compute per-cell marker scores  
- Compute per-cluster mean marker expression  
- Auto-suggest labels while keeping annotation human-guided  
- Works with your custom marker dictionaries

---

## 📦 Installation

Clone and install locally:

```bash
pip install -e .
