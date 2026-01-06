"""
Preprocessing utilities: QC, filtering, Scrublet, HVGs, normalization, PCA.

The main function here is `preprocess_to_pca`, which prepares an AnnData object
up to the PCA step, without performing any batch correction or clustering.

Key design choice (infercnv-friendly):
- Keep ALL genes (do NOT subset to HVGs)
- Use HVGs only for PCA/embedding (mask via `use_highly_variable=True`)
- Keep normalized+log1p ALL genes available via `.raw` and `layers["log1p"]`

Update requested:
- Make cell QC filtering **batch-specific** (per batch thresholds), while keeping
  Scrublet exactly as-is (using scanpy's batch_key support if provided).

Also included (highly recommended, minimal complexity):
- Avoid creating a dense full-gene `scaled_hvg` layer (memory risk). Instead,
  scale HVGs and run PCA on the HVG matrix only (sklearn), then store results in
  `adata.obsm["X_pca"]` + `adata.uns["pca"]`. This keeps the rest of your pipeline
  unchanged because downstream steps typically use `X_pca`.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import scanpy as sc
from anndata import AnnData


def _per_batch_qc_filter(
    adata: AnnData,
    *,
    batch_key: str,
    # absolute floors
    min_genes: int,
    min_counts: int,
    # absolute caps (can be np.inf)
    max_genes: int,
    max_counts: float,
    # optional qc caps (if None -> use quantile cap)
    use_mito_filter: bool,
    max_pct_mito: float | None,
    use_ribo_filter: bool,
    max_pct_ribo: float | None,
    use_hb_filter: bool,
    max_pct_hb: float | None,
    # per-batch quantile caps (robust defaults)
    max_genes_q: float = 0.995,
    max_counts_q: float = 0.995,
    max_pct_mito_q: float = 0.95,
    max_pct_ribo_q: float = 0.95,
    max_pct_hb_q: float = 0.95,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, Dict[str, float | int | None]]]:
    """
    Return:
      - global boolean mask for cells to keep
      - dict of per-batch thresholds used (for reproducibility/debugging)

    Policy:
      - Floors (min_genes/min_counts) are fixed across batches.
      - Caps (max_genes/max_counts) are tightened per batch using quantiles,
        then also tightened by user-provided caps (take min).
      - For pct_* caps:
          * if user cap is not None: use min(user_cap, per-batch-quantile)
          * if user cap is None: use per-batch-quantile (i.e., adaptive)
    """
    if batch_key not in adata.obs:
        raise KeyError(f"batch_key={batch_key!r} not found in adata.obs")

    batch_vals = adata.obs[batch_key].astype(str)
    mask = np.zeros(adata.n_obs, dtype=bool)

    thresholds: Dict[str, Dict[str, float | int | None]] = {}

    def _tight_cap(user_cap: float | None, q_cap: float) -> float:
        return float(q_cap) if user_cap is None else float(min(user_cap, q_cap))

    for b in batch_vals.unique():
        idx = (batch_vals == b).to_numpy()
        sub = adata.obs.iloc[np.where(idx)[0]]

        # quantile-derived caps for genes/counts
        q_max_genes = float(np.quantile(sub["n_genes_by_counts"].to_numpy(), max_genes_q))
        q_max_counts = float(np.quantile(sub["total_counts"].to_numpy(), max_counts_q))

        cap_genes = int(min(max_genes, q_max_genes)) if np.isfinite(max_genes) else int(q_max_genes)
        cap_counts = float(min(max_counts, q_max_counts)) if np.isfinite(max_counts) else float(q_max_counts)

        # pct caps (only if those columns exist)
        cap_mito = None
        if use_mito_filter and "pct_counts_mt" in sub.columns:
            q = float(np.quantile(sub["pct_counts_mt"].to_numpy(), max_pct_mito_q))
            cap_mito = _tight_cap(max_pct_mito, q)

        cap_ribo = None
        if use_ribo_filter and "pct_counts_ribo" in sub.columns:
            q = float(np.quantile(sub["pct_counts_ribo"].to_numpy(), max_pct_ribo_q))
            cap_ribo = _tight_cap(max_pct_ribo, q)

        cap_hb = None
        if use_hb_filter and "pct_counts_hb" in sub.columns:
            q = float(np.quantile(sub["pct_counts_hb"].to_numpy(), max_pct_hb_q))
            cap_hb = _tight_cap(max_pct_hb, q)

        thresholds[str(b)] = {
            "min_genes": int(min_genes),
            "max_genes": int(cap_genes),
            "min_counts": int(min_counts),
            "max_counts": float(cap_counts),
            "max_pct_mito": cap_mito,
            "max_pct_ribo": cap_ribo,
            "max_pct_hb": cap_hb,
        }

        m = (
            (sub["n_genes_by_counts"] >= min_genes)
            & (sub["n_genes_by_counts"] <= cap_genes)
            & (sub["total_counts"] >= min_counts)
            & (sub["total_counts"] <= cap_counts)
        )

        if cap_mito is not None and "pct_counts_mt" in sub.columns:
            m &= sub["pct_counts_mt"] <= cap_mito
        if cap_ribo is not None and "pct_counts_ribo" in sub.columns:
            m &= sub["pct_counts_ribo"] <= cap_ribo
        if cap_hb is not None and "pct_counts_hb" in sub.columns:
            m &= sub["pct_counts_hb"] <= cap_hb

        mask[np.where(idx)[0]] = m.to_numpy()

        if verbose:
            kept = int(m.sum())
            total = int(m.shape[0])
            msg = (
                f"[QC filter] batch={b} kept {kept}/{total} | "
                f"max_genes={cap_genes}, max_counts={cap_counts:.0f}"
            )
            if cap_mito is not None:
                msg += f", max_mito={cap_mito:.2f}"
            print(msg)

    return mask, thresholds


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
    # NEW: batch-specific filtering knobs (simple + robust)
    qc_per_batch: bool = True,
    max_genes_q: float = 0.995,
    max_counts_q: float = 0.995,
    max_pct_mito_q: float = 0.95,
    max_pct_ribo_q: float = 0.95,
    max_pct_hb_q: float = 0.95,
    # whether to store normalized log-expression in .raw (snapshot)
    set_raw: bool = True,
    # Scrublet doublet removal (UNCHANGED)
    run_scrublet: bool = True,
    scrublet_sim_doublet_ratio: float = 2.0,
    scrublet_n_neighbors: int = 30,
    scrublet_threshold: float = 0.25,
    # library-size normalization
    target_sum: float = 1e4,
    # PCA
    n_pcs: int = 50,
    scale_max_value: float = 10.0,
    random_state: int = 0,
    # layers
    log1p_layer_key: str = "log1p",
    verbose: bool = True,
) -> AnnData:
    """
    Preprocess scRNA-seq data up to PCA (no integration, no clustering).

    Pipeline:
      0) Ensure raw counts are available in `adata.layers["counts"]` and use counts in `.X`
      1) Filter genes by minimum number of cells (on counts)
      2) Compute QC metrics (mito/ribo/hb)
      3) Filter cells by QC thresholds
           - if qc_per_batch=True and batch_key is provided -> per-batch adaptive caps
           - else -> global fixed thresholds (original behavior)
      4) Run Scrublet (UNCHANGED; uses scanpy scrublet with optional batch_key)
      5) Normalize + log1p ALL genes (store in `layers["log1p"]`)
      6) Compute HVGs (mask stored in `adata.var["highly_variable"]`, no subsetting)
      7) PCA on scaled HVG matrix (memory-safe; no dense full-gene layer)

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
        # Assume X is counts if counts layer missing
        adata.layers["counts"] = adata.X
        adata.X = adata.layers["counts"]

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
    # 3) Cell-level filters (batch-specific if enabled)
    # ------------------------------------------------------------------ #
    if qc_per_batch and batch_key is not None:
        cell_filter, per_batch_thresholds = _per_batch_qc_filter(
            adata,
            batch_key=batch_key,
            min_genes=min_genes,
            max_genes=max_genes,
            min_counts=min_counts,
            max_counts=max_counts,
            use_mito_filter=use_mito_filter,
            max_pct_mito=max_pct_mito,
            use_ribo_filter=use_ribo_filter,
            max_pct_ribo=max_pct_ribo,
            use_hb_filter=use_hb_filter,
            max_pct_hb=max_pct_hb,
            max_genes_q=max_genes_q,
            max_counts_q=max_counts_q,
            max_pct_mito_q=max_pct_mito_q,
            max_pct_ribo_q=max_pct_ribo_q,
            max_pct_hb_q=max_pct_hb_q,
            verbose=verbose,
        )
    else:
        per_batch_thresholds = {}

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

    n_before = adata.n_obs
    adata = adata[cell_filter, :].copy()

    # Keep layers consistent after slicing (avoid views/backing surprises)
    if "counts" in adata.layers:
        adata.layers["counts"] = adata.layers["counts"].copy()

    if verbose:
        print(f"[QC filter] removed {n_before - adata.n_obs} cells; remaining {adata.n_obs}")

    # Store thresholds used (helpful for reproducibility)
    adata.uns.setdefault("scrna_pipeline", {})
    adata.uns["scrna_pipeline"].setdefault("preprocess_to_pca", {})
    adata.uns["scrna_pipeline"]["preprocess_to_pca"]["qc_per_batch"] = bool(qc_per_batch and batch_key is not None)
    adata.uns["scrna_pipeline"]["preprocess_to_pca"]["qc_thresholds_per_batch"] = per_batch_thresholds

    # ------------------------------------------------------------------ #
    # 4) Scrublet doublet detection (UNCHANGED; on counts)
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
    if "counts" not in adata.layers:
        raise RuntimeError("Expected `adata.layers['counts']` to exist before normalization.")

    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    adata.layers[log1p_layer_key] = adata.X.copy()

    if set_raw:
        # snapshot of log1p normalized all-genes expression
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

    hvgs = adata.var["highly_variable"].to_numpy()
    n_hvg = int(hvgs.sum())
    if n_hvg == 0:
        raise RuntimeError("No HVGs were selected (adata.var['highly_variable'].sum() == 0).")

    # ------------------------------------------------------------------ #
    # 7) PCA on scaled HVGs only (memory-safe; no dense full-gene layer)
    # ------------------------------------------------------------------ #
    # Keep `.X` as log1p all genes (canonical expression view)
    # Compute PCA from scaled HVG matrix and write result to `obsm["X_pca"]`.
    try:
        from scipy import sparse
        from sklearn.decomposition import PCA
    except Exception as e:
        raise ImportError("This PCA path requires scipy and scikit-learn.") from e

    X_log1p = adata.layers[log1p_layer_key]
    X_hvg = X_log1p[:, hvgs]
    X_hvg = X_hvg.toarray() if sparse.issparse(X_hvg) else np.asarray(X_hvg)

    # standardize HVGs
    mu = X_hvg.mean(axis=0, keepdims=True)
    sd = X_hvg.std(axis=0, ddof=0, keepdims=True)
    sd[sd == 0] = 1.0
    X_hvg = (X_hvg - mu) / sd
    X_hvg = np.clip(X_hvg, -scale_max_value, scale_max_value)

    pca = PCA(n_components=n_pcs, random_state=random_state)
    adata.obsm["X_pca"] = pca.fit_transform(X_hvg)

    # scanpy-friendly pca metadata
    adata.uns.setdefault("pca", {})
    adata.uns["pca"]["variance_ratio"] = pca.explained_variance_ratio_
    adata.uns["pca"]["params"] = {
        "n_pcs": n_pcs,
        "scale_max_value": scale_max_value,
        "source_layer": log1p_layer_key,
        "hvg_mask": True,
    }

    return adata
