"""
Preprocessing utilities: QC, filtering, Scrublet, HVGs, normalization, PCA.

The main function here is `preprocess_to_pca`, which prepares an AnnData object
up to the PCA step, without performing any batch correction or clustering.

Typical downstream usage:
    - apply a batch correction method using `apply_batch_correction`
    - run neighbors / clustering / UMAP using `cluster_and_embed`
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
    max_pct_mito: float | None = 10.0,    # default 10%; set None to disable threshold
    use_ribo_filter: bool = True,
    max_pct_ribo: float | None = 10.0,    # default 10%; set None to disable threshold
    use_hb_filter: bool = False,
    max_pct_hb: float | None = 1.0,       # default 1%; set None to disable threshold
    # whether to store normalized log-expression in .raw
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
    verbose: bool = True,
) -> AnnData:
    """
    Preprocess scRNA-seq data up to PCA.

    This function performs the "early" steps that are common to most analyses:

        0) Ensure raw counts are available in `adata.layers["counts"]`.
        1) Filter genes by minimum number of cells.
        2) Compute QC metrics (mito / ribo / hemoglobin percentages).
        3) Optionally filter cells based on:
           - gene counts
           - total counts
           - percent mito / ribo / hemoglobin
        4) (Optional) Run Scrublet to remove predicted doublets.
        5) Select highly variable genes (HVGs) with correct ordering:
           - If `hvg_flavor == "seurat_v3"`:
               HVGs are computed on raw counts (`layer="counts"`),
               then normalize/log, optionally store `.raw`, then subset to HVGs.
           - Otherwise:
               normalize + log first, optionally store `.raw`,
               then HVGs on log-normalized X, then subset to HVGs.
        6) Scale and run PCA, storing the result in `adata.obsm["X_pca"]`.

    `adata.raw` behavior
    --------------------
    If `set_raw=True` (default), the function stores a **normalized, log-transformed
    version with all genes** in `adata.raw` *before* subsetting to HVGs. This is
    the recommended pattern for downstream DEG analysis and marker scoring
    (e.g. `sc.tl.rank_genes_groups`, `sc.tl.score_genes` with `use_raw=True`).

    For very large datasets, users can disable this behavior with `set_raw=False`
    to save memory.

    QC filters
    ----------
    All three QC filters (mito / ribo / hb) are optional:
        - `use_*_filter` controls whether the filter is applied.
        - If a filter is enabled but its `max_pct_*` is None, a standard default
          value is used (mito=10%, ribo=10%, hb=1%).
    """

    # ------------------------------------------------------------------ #
    # 0) Ensure we are working with RAW COUNTS
    # ------------------------------------------------------------------ #
    # We keep a copy of the raw counts in `layers["counts"]` and also
    # ensure that X contains counts before starting QC and HVG selection.
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    else:
        adata.layers["counts"] = adata.X.copy()

    # ------------------------------------------------------------------ #
    # 1) Basic gene filter (remove genes seen in very few cells)
    # ------------------------------------------------------------------ #
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)

    # ------------------------------------------------------------------ #
    # 2) QC metrics: mitochondrial / ribosomal / hemoglobin genes
    # ------------------------------------------------------------------ #
    qc_vars: List[str] = []

    # Mitochondrial genes: typically "MT-" prefix in human.
    # We always compute mito QC so users can inspect it, even if they
    # choose not to filter later.
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    qc_vars.append("mt")

    # Ribosomal genes: RPL* and RPS* (if enabled)
    if use_ribo_filter:
        upper_names = adata.var_names.str.upper()
        adata.var["ribo"] = (
            upper_names.str.startswith("RPL") |
            upper_names.str.startswith("RPS")
        )
        qc_vars.append("ribo")

    # Hemoglobin genes: HBA*, HBB*, HBD*, HBE*, HBG* (if enabled)
    if use_hb_filter:
        upper_names = adata.var_names.str.upper()
        hb_prefixes = ("HBA", "HBB", "HBD", "HBE", "HBG")
        adata.var["hb"] = np.logical_or.reduce(
            [upper_names.str.startswith(pref) for pref in hb_prefixes]
        )
        qc_vars.append("hb")

    # Compute per-cell QC metrics:
    #   - total_counts
    #   - n_genes_by_counts
    #   - pct_counts_<qc_var> for each entry in qc_vars
    sc.pp.calculate_qc_metrics(adata, qc_vars=qc_vars, inplace=True)

    # ------------------------------------------------------------------ #
    # 3) Cell-level filters
    # ------------------------------------------------------------------ #
    # Basic filters on number of genes and total counts
    cell_filter = (
        (adata.obs["n_genes_by_counts"] >= min_genes) &
        (adata.obs["n_genes_by_counts"] <= max_genes) &
        (adata.obs["total_counts"] >= min_counts) &
        (adata.obs["total_counts"] <= max_counts)
    )

    # Mitochondrial content filter (optional)
    if use_mito_filter and "pct_counts_mt" in adata.obs:
        threshold = 10.0 if max_pct_mito is None else max_pct_mito
        cell_filter &= adata.obs["pct_counts_mt"] <= threshold

    # Ribosomal content filter (optional)
    if use_ribo_filter and "pct_counts_ribo" in adata.obs:
        threshold = 10.0 if max_pct_ribo is None else max_pct_ribo
        cell_filter &= adata.obs["pct_counts_ribo"] <= threshold

    # Hemoglobin content filter (optional)
    if use_hb_filter and "pct_counts_hb" in adata.obs:
        threshold = 1.0 if max_pct_hb is None else max_pct_hb
        cell_filter &= adata.obs["pct_counts_hb"] <= threshold

    # Apply all filters at once
    adata = adata[cell_filter, :].copy()

    # ------------------------------------------------------------------ #
    # 4) Scrublet-based doublet detection (on raw counts)
    # ------------------------------------------------------------------ #
    if run_scrublet:
        if verbose:
            print("Running Scrublet for doublet detection...")

        scrublet_kwargs = dict(
            sim_doublet_ratio=scrublet_sim_doublet_ratio,
            n_neighbors=scrublet_n_neighbors,
            threshold=scrublet_threshold,
        )
        # Only pass batch_key to Scrublet if it was provided
        if batch_key is not None:
            scrublet_kwargs["batch_key"] = batch_key

        sc.pp.scrublet(adata, **scrublet_kwargs)

        n_before = adata.n_obs
        adata = adata[~adata.obs["predicted_doublet"], :].copy()
        n_after = adata.n_obs

        if verbose:
            print(f"Removed {n_before - n_after} predicted doublets")
            print("After doublet removal:", adata.shape)

    # ------------------------------------------------------------------ #
    # 5 & 6) HVGs + normalization / log1p + (optional) .raw
    # ------------------------------------------------------------------ #
    if hvg_flavor == "seurat_v3":
        # --- HVGs on RAW COUNTS (Seurat v3-style) ---
        # Uses counts from `layer="counts"` and can use batch_key for
        # per-batch HVG selection.
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor="seurat_v3",
            batch_key=batch_key,
            layer="counts",   # crucial: HVGs computed on raw counts
        )

        # --- Then normalize & log-transform ALL genes ---
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)

        # Save clean normalized matrix for downstream DEG / scoring
        if set_raw:
            # Note: this duplicates the expression matrix and can be
            # memory-intensive for very large datasets.
            adata.raw = adata.copy()

        # Finally, subset to HVGs selected from counts
        adata = adata[:, adata.var["highly_variable"]].copy()

    else:
        # --- First normalize & log-transform ALL genes ---
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)

        # Save clean normalized matrix before HVG filtering
        if set_raw:
            adata.raw = adata.copy()

        # --- Then compute HVGs on log-normalized X ---
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor=hvg_flavor,
            batch_key=batch_key,

        )
        adata = adata[:, adata.var["highly_variable"]].copy()

    # ------------------------------------------------------------------ #
    # 7) Scale + PCA
    # ------------------------------------------------------------------ #
    sc.pp.scale(adata, max_value=scale_max_value)
    sc.tl.pca(
        adata,
        n_comps=n_pcs,
        svd_solver="arpack",
        random_state=random_state,
    )

    return adata
