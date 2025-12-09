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

---

### **4. Marker-Based Annotation**
- Auto-suggested labels   
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
git clone https://github.com/Irfanwustl/scRNA-seq-pipeline.git
cd scRNA-seq-pipeline
```

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
pip install ".[pip_extras]"
```

---

### ✔️ After Installation

Verify installation:

```bash
python -c "import scrna_pipeline; print('Installed successfully!')"
```

---

## 🧪 Quick Example: From RAW Matrix → h5ad → Full Pipeline → Annotation

```python
import scanpy as sc
from scrna_pipeline.h5ad_builders import raw_matrix_to_h5ad
from scrna_pipeline import standard_scrna_pipeline
from scrna_pipeline.annotation import score_markers_and_suggest_labels

# ---------------------------------------------------------------
# 1) Convert a GEO-style RAW matrix into an AnnData (.h5ad) file
# ---------------------------------------------------------------
# Adjust the paths according to your dataset structure
raw_txt = "input.raw.matrix.txt.gz"
output_h5ad = "output.h5ad"

adata_raw = raw_matrix_to_h5ad(
    raw_file=raw_txt,
    genes_are_rows=True,          # typical for GEO RAW matrices
    sample_from_obs_names="suffix",
    output_h5ad=output_h5ad,
)

# Load the created AnnData file
adata_raw = sc.read_h5ad(output_h5ad)

# ---------------------------------------------------------------
# 2) Run the standard pipeline 
# ---------------------------------------------------------------

# If your dataset has no batches, set:
#     batch_key=None, batch_method="none"
batch_key = "sample"

adata_proc = standard_scrna_pipeline(
    adata_raw,
    batch_key=batch_key,
    batch_method="harmony",     # use “none” to disable batch correction

    # HVG settings
    hvg_flavor="seurat_v3",
    n_top_genes=2000,

    # Clustering
    clustering_method="leiden",

    # Extra preprocessing options
    preprocess_kwargs={
        "use_mito_filter": True,
        "use_ribo_filter": False,
    },
)

# ---------------------------------------------------------------
# 3) Marker-based annotation
# ---------------------------------------------------------------

# Example marker dictionary (edit for your tissues)
marker_dict = {
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19"],
    "T_cell": ["CD3D", "CD3E", "CD2", "CD8A", "CD4"],
    "B_cell": ["MS4A1", "CD79A", "CD79B"],
    "Myeloid": ["LYZ", "S100A8", "S100A9"],
    "Endothelial": ["PECAM1", "VWF", "KDR"],
    "Fibroblast": ["COL1A1", "COL1A2", "DCN"],
    "Mast": ["TPSAB1", "TPSB2", "CPA3"],
}

# Pick clustering column automatically
cluster_key = (
    "leiden" if "leiden" in adata_proc.obs else
    "louvain" if "louvain" in adata_proc.obs else
    "cluster" if "cluster" in adata_proc.obs else
    None
)

if cluster_key is None:
    raise ValueError("No clustering column found in adata.obs.")

# Compute cluster-level marker scores + suggested labels
cluster_scores, suggested_labels = score_markers_and_suggest_labels(
    adata_proc,
    marker_dict,
    cluster_key=cluster_key,
)

# Store suggested cell type labels
adata_proc.obs["celltype"] = adata_proc.obs[cluster_key].map(suggested_labels)

# ---------------------------------------------------------------
# 4) Plot UMAP with suggested cell types
# ---------------------------------------------------------------
sc.pl.umap(
    adata_proc,
    color=["celltype", cluster_key],
    legend_loc="on data",
    frameon=False,
    title="Marker-Based Cell Type Annotation",
    legend_fontsize=8,
)
```

For details, look at the example folder.

---


## 📄 License

MIT License. See the [LICENSE](LICENSE) file in the repository root for full details.

