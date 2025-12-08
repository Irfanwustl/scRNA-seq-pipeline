# scRNA-seq Pipeline

A clean, modular, and extensible **scRNA-seq analysis pipeline** written in Python.  
This package provides a standard **end-to-end workflow**—from QC and Scrublet to HVGs, PCA, batch correction, clustering, and UMAP—built on top of Scanpy but designed for clarity, reproducibility, and extensibility.

---

## 🚀 Features

### **1. Preprocessing**
- Gene & cell QC filtering  
- Mito / ribosomal / hemoglobin content filtering  
- Optional **Scrublet** doublet removal  
- Highly Variable Gene (HVG) selection  
- Library-size normalization and log1p transformation   

---

### **2. Batch Correction**
- **Harmony** integration (default)  
- Or disable with `batch_method="none"`

---

### **3. Clustering & Embedding**
- KNN graph construction  
- **Leiden** or **Louvain** clustering  
- UMAP embedding (stored in `adata.obsm["X_umap"]`)

---

### **4. Marker-Based Annotation**
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

### ✔️ This package supports **Approach B**
If you provide a single AnnData object containing multiple samples (with `adata.obs["sample"]`), the pipeline will treat them as one dataset and optionally apply **Harmony** to remove sample effects.

If you prefer Approach A, simply run your own per-sample preprocessing and merge the resulting AnnData objects before passing them to downstream steps.

---

## 📦 Installation

There are three clean ways to install this package depending on your workflow.

---

### **1️⃣ Install using pip (recommended for most users)**

Clone the repository:

```bash
git clone https://github.com/<your-username>/scrna-seq-pipeline.git
cd scrna-seq-pipeline
````

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the pipeline itself:

```bash
pip install -e .
```

You can now import the pipeline:

```python
from scrna_pipeline import standard_scrna_pipeline
```

---

### **2️⃣ Install using Conda (best for full reproducibility)**

Create the environment:

```bash
conda env create -f environment.yml
conda activate scrna-pipeline
```

Then install the package:

```bash
pip install -e .
```

---

### **3️⃣ (Optional) Install pip-only extras**

Some tools such as `scrublet`, `harmonypy`, and `igraph` may require pip-based wheels.
If needed:

```bash
pip install .[pip_extras]
```

---

### ✔️ After Installation

Verify installation:

```bash
python -c "import scrna_pipeline; print('Installed successfully!')"
```

---

### 📝 Requirements

* Python 3.10+
* Works on macOS (Intel/M1/M2/M3), Linux, and cloud environments

---

## 🧪 Quickstart Example

```python
import scanpy as sc
from scrna_pipeline import standard_scrna_pipeline

# Load your raw or processed AnnData
adata = sc.read_h5ad("input.h5ad") # to build h5ad file use functions h5ad_builders.py

# Run the pipeline
adata_proc = standard_scrna_pipeline(
    adata,
    sample_key="sample",
    batch_method="harmony",     # or "none"
    min_genes=500,
    min_cells=3,
    scrublet=True
)

# Inspect UMAP
sc.pl.umap(adata_proc, color=["louvain", "suggested_label"])
```

For details, look at the example folder.

---


## 📄 License

MIT License. See the [LICENSE](LICENSE) file in the repository root for full details.

