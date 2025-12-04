# scRNA-seq Pipeline

A clean, modular, and extensible **scRNA-seq analysis pipeline** written in Python.  
This package provides a standard **end-to-end workflow**—from QC and Scrublet to HVGs, PCA, batch correction, clustering, and UMAP—built on top of Scanpy but designed for clarity, reproducibility, and extensibility.

---

## 🚀 Features

### 1. Preprocessing
- Gene & cell QC filtering  
- Mito / ribosomal / hemoglobin content filtering  
- Optional **Scrublet** doublet removal  
- Highly Variable Gene (HVG) selection  
  - Default: **Seurat v3 HVGs on raw counts**  
- Library-size normalization and log1p transformation  
- Optional `.raw` storage of normalized data  
- Scaling + PCA (stored in `adata.obsm["X_pca"]`)

---

### 2. Batch Correction
- **Harmony** integration (default)  
- Or disable with `batch_method="none"`

---

### 3. Clustering & Embedding
- KNN graph construction  
- **Leiden** or **Louvain** clustering  
- UMAP embedding (stored in `adata.obsm["X_umap"]`)

---

### 4. Marker-Based Annotation
- Per-cell marker scoring  
- Cluster-level mean marker expression  
- Auto-suggested labels (human-guided, not blind)  
- Easily plug in your own marker dictionaries

---

## 📘 Working With Multi-Sample Datasets

There are two approaches in scRNA-seq:

### **A. Preprocess each sample independently (then merge)**
Use this when:
- samples differ in sequencing depth, chemistry, or library prep  
- you want sample-specific QC thresholds  
- file sizes are large and easier to handle per-sample  

### **B. Merge raw counts first, then preprocess together**
Use when:
- all samples follow the same protocol  
- you have enough memory to hold a combined matrix  
- you want unified HVG selection and PCA across all samples  

### **This package supports B now**  
If you provide a single AnnData object containing multiple samples (with `adata.obs["sample"]`), the pipeline will treat them as one dataset and optionally apply **Harmony** to remove sample effects.

If you prefer approach A, simply run your own per-sample preprocessing and merge the resulting AnnData objects before passing them to downstream steps.

---

## 📦 Installation

Clone and install locally:

```bash
pip install -e .
