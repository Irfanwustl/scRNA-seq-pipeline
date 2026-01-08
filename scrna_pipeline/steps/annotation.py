"""
Helpers for semi-automatic cell type annotation.

Includes:
- marker-based scoring and label suggestion per cluster
- per-batch broad cell type annotation (no integration)

These utilities are NOT meant to replace manual annotation.
Always validate with marker plots and biological knowledge.
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
    cluster_key: str = "leiden",
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

        Only genes present in the scoring gene space (adata.raw.var_names if use_raw, else adata.var_names) will be used, so you can safely
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
    var_names = adata.raw.var_names if use_raw_eff else adata.var_names

    score_cols: List[str] = []
    for celltype, genes in marker_dict.items():
        genes_use = [g for g in genes if g in var_names]
        if not genes_use:
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
        src = "adata.raw.var_names" if use_raw_eff else "adata.var_names"
        raise ValueError(
            f"No marker genes from marker_dict found in {src}. "
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


from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from anndata import AnnData


@dataclass(frozen=True)
class BroadAnnotationStep:
    """
    Broad (compartment-level) cluster-based annotation.

    Contract
    --------
    Requires:
      - adata.obs[cluster_key] exists (so run clustering first)
    Writes:
      - adata.obs[broad_label_key] as category
      - adata.uns["scrna_pipeline"]["broad_annotation"] summary
    """
    name: str = "broad_annotation"

    marker_dict: Dict[str, List[str]] | None = None
    cluster_key: str = "leiden"
    broad_label_key: str = "broad_celltype"
    score_prefix: str = "broad_"
    use_raw: bool | None = None

    # Escape hatch for future options without changing the signature a lot
    params: Optional[Dict[str, Any]] = None

    def run(self, adata: AnnData, ctx) -> AnnData:
        if self.marker_dict is None:
            raise ValueError("BroadAnnotationStep requires marker_dict (got None).")

        if self.cluster_key not in adata.obs:
            raise KeyError(
                f"cluster_key={self.cluster_key!r} not found in adata.obs. "
                "Run clustering first (e.g., ClusterAndEmbedStep)."
            )

        kwargs = dict(self.params or {})
        cluster_scores, suggested = score_markers_and_suggest_labels(
            adata,
            self.marker_dict,
            cluster_key=self.cluster_key,
            score_prefix=self.score_prefix,
            use_raw=self.use_raw,
            **kwargs,
        )

        adata.obs[self.broad_label_key] = (
            adata.obs[self.cluster_key].map(suggested).astype("category")
        )

        # store summary for inspection/reproducibility
        adata.uns.setdefault("scrna_pipeline", {})
        adata.uns["scrna_pipeline"]["broad_annotation"] = {
            "label_key": self.broad_label_key,
            "cluster_key": self.cluster_key,
            "score_prefix": self.score_prefix,
            "use_raw": self.use_raw,
            "cluster_scores_shape": list(cluster_scores.shape),
        }

        return adata

    def outputs(self) -> Tuple[str, ...]:
        return (f"obs[{self.broad_label_key}]",)




## below is for multi-sample adata so Delete the below?
# def annotate_broad_celltypes_per_batch(
#     adata: AnnData,
#     marker_dict: Dict[str, List[str]],
#     *,
#     batch_key: str = "sample",
#     out_key: str = "broad_celltype",
#     # lightweight per-batch clustering config (non-integrated)
#     rep_key: str = "X_pca",
#     n_neighbors: int = 15,
#     leiden_resolution: float = 0.6,
#     cluster_key: str = "_tmp_leiden",
#     # scoring config
#     score_prefix: str = "broad_",
#     use_raw: bool | None = None,
#     verbose: bool = True,
# ) -> Tuple[pd.DataFrame, pd.Series]:
#     """
#     Assign broad cell types PER BATCH, without batch correction.

#     Strategy:
#       - For each batch: build neighbors on rep_key (default X_pca), Leiden cluster
#       - Use your existing `score_markers_and_suggest_labels` to label clusters
#       - Write per-cell labels into `adata.obs[out_key]`

#     Returns
#     -------
#     all_cluster_scores : pd.DataFrame
#         MultiIndex rows (batch, cluster) with mean marker scores per cluster.

#     all_suggested : pd.Series
#         MultiIndex (batch, cluster) -> suggested label.
#     """
#     if batch_key not in adata.obs:
#         raise KeyError(f"batch_key={batch_key!r} not in adata.obs")
#     if rep_key not in adata.obsm:
#         raise KeyError(f"rep_key={rep_key!r} not in adata.obsm. Run preprocess_to_pca first.")

#     adata.obs[out_key] = "Unknown"

#     batch_vals = adata.obs[batch_key].astype(str)
#     batches = batch_vals.unique().tolist()

#     cluster_scores_list: list[pd.DataFrame] = []
#     suggested_list: list[pd.Series] = []

#     for b in batches:
#         idx = (batch_vals == b).to_numpy()
#         if idx.sum() == 0:
#             continue

#         sub = adata[idx].copy()

#         # ---- quick within-batch clustering on non-integrated representation ----
#         sc.pp.neighbors(sub, use_rep=rep_key, n_neighbors=n_neighbors)
#         sc.tl.leiden(sub, resolution=leiden_resolution, key_added=cluster_key)

#         # ---- your existing scoring + suggestions ----
#         cluster_scores, suggested = score_markers_and_suggest_labels(
#             sub,
#             marker_dict,
#             cluster_key=cluster_key,
#             score_prefix=score_prefix,
#             use_raw=use_raw,
#         )

#         # Map cluster -> label and write per-cell labels back
#         sub_labels = sub.obs[cluster_key].map(suggested).astype(str)
#         adata.obs.loc[adata.obs.index[idx], out_key] = sub_labels.to_numpy()

#         # collect outputs for reporting
#         cs = cluster_scores.copy()
#         cs.index = pd.MultiIndex.from_product([[b], cs.index.astype(str)], names=[batch_key, "cluster"])
#         cluster_scores_list.append(cs)

#         sug = suggested.copy()
#         sug.index = pd.MultiIndex.from_product([[b], sug.index.astype(str)], names=[batch_key, "cluster"])
#         suggested_list.append(sug)

#         if verbose:
#             vc = pd.Series(sub_labels).value_counts().to_dict()
#             print(f"[broad annotate] batch={b} -> {vc}")

#     all_cluster_scores = pd.concat(cluster_scores_list, axis=0) if cluster_scores_list else pd.DataFrame()
#     all_suggested = pd.concat(suggested_list, axis=0) if suggested_list else pd.Series(dtype=str)

#     return all_cluster_scores, all_suggested

