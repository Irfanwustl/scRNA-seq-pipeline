"""
Preprocessing utilities: QC, filtering, Scrublet, HVGs, normalization, PCA.

The main function here is `preprocess_to_pca`, which prepares an AnnData object
up to the PCA step, without performing any batch correction or clustering.

Key design choice (infercnv-friendly):
- Keep ALL genes (do NOT subset to HVGs)
- Use HVGs only for PCA/embedding (mask via `use_highly_variable=True`)
- Keep normalized+log1p ALL genes available via `.raw` and `layers["log1p"]`
"""

from __future__ import annotations

from typing import List

import numpy as np
import scanpy as sc
from anndata import AnnData


def preprocess_to_pca(
    adata: AnnData,
    *,
    batch_key: str | None = "sample",
    hvg_flavor: str = "seurat_v3",        # "seurat_v3" (on counts) or e.g. "seurat"
    n_top_genes: int = 2000,
    # gene / cell filter thresholds
    min_cells_per_gene: int = 3,
    min_genes: int = 300,
    max_genes: int = 7000,
    min_counts: int = 500,
    max_counts: float = np.inf,
    # optional QC-based filters
    use_mito_filter: bool = True,
    max_pct_mito: float | None = 10.0,
    use_ribo_filter: bool = True,
    max_pct_ribo: float | None = 10.0,
    use_hb_filter: bool = False,
    max_pct_hb: float | None = 1.0,
    # whether to store normalized log-expression in .raw (snapshot)
    set_raw: bool = True,
    # Scrublet doublet removal
    run_scrublet: bool = True,
    scrublet_sim_doublet_ratio: float = 2.0,
    scrublet_n_neighbors: int = 30,
    scrublet_threshold: float = 0.25,
    # library-size normalization
    target_sum: float = 1e4,
    # scaling / PCA
    n_pcs: int = 50,
    scale_max_value: float = 10.0,
    random_state: int = 0,
    # whether to store scaled HVGs in a layer (recommended)
    store_scaled_hvg_layer: bool = True,
    scaled_hvg_layer_key: str = "scaled_hvg",
    log1p_layer_key: str = "log1p",
    verbose: bool = True,
) -> AnnData:
    """
    Preprocess scRNA-seq data up to PCA.

    Pipeline:
      0) Ensure raw counts are available in `adata.layers["counts"]` and use counts in `.X`
      1) Filter genes by minimum number of cells (on counts)
      2) Compute QC metrics (mito/ribo/hb)
      3) Filter cells by QC thresholds (on counts/QC metrics)
      4) Run Scrublet (on counts)
      5) Normalize + log1p ALL genes (store in `layers["log1p"]`)
      6) Compute HVGs (mask stored in `adata.var["highly_variable"]`, no subsetting)
      7) Scale ONLY HVGs (store scaled HVGs in `layers["scaled_hvg"]` if enabled)
      8) PCA using HVG mask (`use_highly_variable=True`)

    infercnvpy-friendly:
      - all genes retained (important for genomic smoothing / CNV patterns)
      - `.raw` stores a snapshot of log1p all genes
    """

    # ------------------------------------------------------------------ #
    # 0) Ensure raw counts layer exists, and `.X` is counts for QC steps
    # ------------------------------------------------------------------ #
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"]
    else:
        adata.layers["counts"] = adata.X

    # ------------------------------------------------------------------ #
    # 1) Basic gene filter (remove genes seen in very few cells)
    # ------------------------------------------------------------------ #
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)

    # ------------------------------------------------------------------ #
    # 2) QC metrics: mitochondrial / ribosomal / hemoglobin genes
    # ------------------------------------------------------------------ #
    qc_vars: List[str] = []

    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    qc_vars.append("mt")

    if use_ribo_filter:
        upper_names = adata.var_names.str.upper()
        adata.var["ribo"] = upper_names.str.startswith("RPL") | upper_names.str.startswith("RPS")
        qc_vars.append("ribo")

    if use_hb_filter:
        upper_names = adata.var_names.str.upper()
        hb_prefixes = ("HBA", "HBB", "HBD", "HBE", "HBG")
        adata.var["hb"] = np.logical_or.reduce([upper_names.str.startswith(pref) for pref in hb_prefixes])
        qc_vars.append("hb")

    sc.pp.calculate_qc_metrics(adata, qc_vars=qc_vars, inplace=True)

    # ------------------------------------------------------------------ #
    # 3) Cell-level filters
    # ------------------------------------------------------------------ #
    cell_filter = (
        (adata.obs["n_genes_by_counts"] >= min_genes)
        & (adata.obs["n_genes_by_counts"] <= max_genes)
        & (adata.obs["total_counts"] >= min_counts)
        & (adata.obs["total_counts"] <= max_counts)
    )

    if use_mito_filter and "pct_counts_mt" in adata.obs:
        thr = 10.0 if max_pct_mito is None else max_pct_mito
        cell_filter &= adata.obs["pct_counts_mt"] <= thr

    if use_ribo_filter and "pct_counts_ribo" in adata.obs:
        thr = 10.0 if max_pct_ribo is None else max_pct_ribo
        cell_filter &= adata.obs["pct_counts_ribo"] <= thr

    if use_hb_filter and "pct_counts_hb" in adata.obs:
        thr = 1.0 if max_pct_hb is None else max_pct_hb
        cell_filter &= adata.obs["pct_counts_hb"] <= thr

    adata = adata[cell_filter, :].copy()

    # Keep layers consistent after slicing (avoid views/backing surprises)
    if "counts" in adata.layers:
        adata.layers["counts"] = adata.layers["counts"].copy()

    # ------------------------------------------------------------------ #
    # 4) Scrublet doublet detection (on counts)
    # ------------------------------------------------------------------ #
    if run_scrublet:
        if verbose:
            print("Running Scrublet for doublet detection...")

        scrublet_kwargs = dict(
            sim_doublet_ratio=scrublet_sim_doublet_ratio,
            n_neighbors=scrublet_n_neighbors,
            threshold=scrublet_threshold,
        )
        if batch_key is not None:
            scrublet_kwargs["batch_key"] = batch_key

        sc.pp.scrublet(adata, **scrublet_kwargs)

        if "predicted_doublet" not in adata.obs:
            raise RuntimeError("Scrublet ran but `adata.obs['predicted_doublet']` is missing.")

        n_before = adata.n_obs
        adata = adata[~adata.obs["predicted_doublet"], :].copy()

        # Keep counts layer consistent after slicing
        if "counts" in adata.layers:
            adata.layers["counts"] = adata.layers["counts"].copy()

        n_after = adata.n_obs

        if verbose:
            print(f"Removed {n_before - n_after} predicted doublets")
            print("After doublet removal:", adata.shape)

    # ------------------------------------------------------------------ #
    # 5) Normalize + log1p ALL genes (store in a layer)
    # ------------------------------------------------------------------ #
    # Work on counts -> normalized/log1p in `.X`, then store it
    if "counts" not in adata.layers:
        raise RuntimeError("Expected `adata.layers['counts']` to exist before normalization.")

    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    # Store log1p all genes (for infercnvpy, gene scoring, DEG, etc.)
    adata.layers[log1p_layer_key] = adata.X.copy()

    # Set .raw as a snapshot of current log1p state
    if set_raw:
        adata.raw = adata

    # ------------------------------------------------------------------ #
    # 6) HVG selection (mask only, NO subsetting)
    # ------------------------------------------------------------------ #
    if hvg_flavor == "seurat_v3":
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor="seurat_v3",
            batch_key=batch_key,
            layer="counts",
        )
    else:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor=hvg_flavor,
            batch_key=batch_key,
        )

    if "highly_variable" not in adata.var.columns:
        raise RuntimeError("HVG computation failed: adata.var['highly_variable'] not found.")

    # ------------------------------------------------------------------ #
    # 7) Scale ONLY HVGs (store to layer), keep `.X` as log1p all-genes
    # ------------------------------------------------------------------ #
    if store_scaled_hvg_layer:
        X0 = adata.X  # log1p all genes
        try:
            sc.pp.scale(
                adata,
                max_value=scale_max_value,
                zero_center=True,
                use_highly_variable=True,
            )
            adata.layers[scaled_hvg_layer_key] = adata.X.copy()
        finally:
            adata.X = X0  # restore log1p all genes

    # ------------------------------------------------------------------ #
    # 8) PCA using HVGs only
    # ------------------------------------------------------------------ #
    if store_scaled_hvg_layer:
        sc.tl.pca(
            adata,
            n_comps=n_pcs,
            svd_solver="arpack",
            random_state=random_state,
            use_highly_variable=True,
            layer=scaled_hvg_layer_key,
        )
    else:
        sc.tl.pca(
            adata,
            n_comps=n_pcs,
            svd_solver="arpack",
            random_state=random_state,
            use_highly_variable=True,
        )

    return adata
