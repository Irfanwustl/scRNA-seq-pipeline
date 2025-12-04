"""
Helpers for semi-automatic cell type annotation.

These functions are NOT meant to replace manual annotation. Instead, they:
- compute marker-based scores per cell and per cluster
- suggest a best-matching cell type per cluster

You should always sanity-check suggestions with marker expression plots
(e.g. dotplot, stacked violin) and biological knowledge before finalizing
annotations.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import scanpy as sc
from anndata import AnnData


def score_markers_and_suggest_labels(
    adata: AnnData,
    marker_dict: Dict[str, List[str]],
    *,
    cluster_key: str = "louvain",
    score_prefix: str = "",
    use_raw: bool | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Compute marker-based scores per cluster and suggest cell type labels.

    Parameters
    ----------
    adata
        AnnData object after clustering (e.g., after your pipeline).
        Must contain `adata.obs[cluster_key]`.

    marker_dict
        Dictionary mapping cell type names to marker gene lists, e.g.:
            {
                "T_cell": ["CD3D", "CD3E", "CD2"],
                "B_cell": ["MS4A1", "CD79A"],
                "Myeloid": ["LYZ", "S100A8"],
                ...
            }

        Only genes present in `adata.var_names` will be used, so you can safely
        pass a large, generic marker list.

    cluster_key
        Observation column that encodes cluster IDs (e.g., "louvain", "leiden").

    score_prefix
        Optional prefix for per-cell score column names in `adata.obs`.
        If non-empty, scores will be stored as f"{score_prefix}{celltype}_score".

    use_raw
        Controls whether to use `adata.raw` for scoring.

        - None (default): use `adata.raw` if it exists, otherwise fall back
          to `adata.X`. This is usually what you want if your preprocessing
          stored a normalized all-genes matrix in `.raw`.
        - True:  require `adata.raw` to be present; raise an informative error
          if it is missing.
        - False: always use `adata.X` for scoring.

    Returns
    -------
    cluster_scores : pd.DataFrame
        DataFrame of shape (n_clusters, n_celltypes) with mean marker scores
        per cluster. Index = cluster IDs, columns = cell type names.

    suggested_labels : pd.Series
        Series mapping cluster ID → suggested cell type
        (argmax over mean scores per cluster).

    Notes
    -----
    - This function does NOT write anything into `adata.obs["celltype"]` by
      default. A typical usage pattern is:

          cluster_scores, suggested = score_markers_and_suggest_labels(
              adata,
              marker_dict,
              cluster_key="louvain",
          )

          # Add suggested labels for visualization
          adata.obs["celltype"] = adata.obs["louvain"].map(suggested)

          sc.pl.umap(adata, color="celltype", legend_loc="on data")

      You should manually inspect `cluster_scores` and the UMAP before
      treating `celltype` as final.
    """
    if cluster_key not in adata.obs:
        raise ValueError(
            f"Cluster key '{cluster_key}' not found in adata.obs. "
            "Did you run clustering?"
        )

    # Decide whether to use .raw or not
    if use_raw is None:
        # Auto-mode: prefer .raw if it exists
        use_raw_eff = adata.raw is not None
    else:
        use_raw_eff = use_raw
        if use_raw and adata.raw is None:
            raise ValueError(
                "use_raw=True was requested, but adata.raw is None. "
                "Either set `adata.raw` during preprocessing, or call "
                "`score_markers_and_suggest_labels(..., use_raw=False)`."
            )

    # ------------------------------------------------------------------ #
    # 1) Per-cell scores for each cell type
    # ------------------------------------------------------------------ #
    score_cols: List[str] = []
    for celltype, genes in marker_dict.items():
        # Filter to genes that exist in this dataset
        genes_use = [g for g in genes if g in adata.var_names]
        if len(genes_use) == 0:
            # Skip if none of the markers are present for this cell type
            continue

        score_name = f"{score_prefix}{celltype}_score"
        score_cols.append(score_name)

        sc.tl.score_genes(
            adata,
            gene_list=genes_use,
            score_name=score_name,
            use_raw=use_raw_eff,
        )

    if not score_cols:
        raise ValueError(
            "No marker genes from marker_dict found in adata.var_names. "
            "Check that your gene symbols match the dataset."
        )

    # ------------------------------------------------------------------ #
    # 2) Average scores per cluster
    # ------------------------------------------------------------------ #
    cluster_scores = (
        adata.obs
        .groupby(cluster_key)[score_cols]
        .mean()
    )

    # ------------------------------------------------------------------ #
    # 3) Suggest labels based on highest mean score per cluster
    # ------------------------------------------------------------------ #
    # Map from score column name back to plain cell type name
    # (remove the score_prefix and trailing "_score").
    rename_map: Dict[str, str] = {}
    for col in score_cols:
        # Example: "T_cell_score" or "basic_T_cell_score" → "T_cell"
        base = col
        if score_prefix and base.startswith(score_prefix):
            base = base[len(score_prefix):]
        if base.endswith("_score"):
            base = base[:-6]
        rename_map[col] = base

    cluster_scores_ctype = cluster_scores.rename(columns=rename_map)

    # For each cluster, pick the cell type with highest mean score
    suggested_labels = cluster_scores_ctype.idxmax(axis=1)

    return cluster_scores_ctype, suggested_labels
