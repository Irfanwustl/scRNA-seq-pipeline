"""
Clustering and UMAP utilities.

This module assumes you already have an embedding in `adata.obsm[rep_key]`
(e.g. PCA or a batch-corrected PCA) and builds:

- a neighborhood graph
- optional graph-based clustering (Leiden or Louvain)
- a UMAP embedding
"""

from __future__ import annotations

from typing import Literal, Optional

import scanpy as sc
from anndata import AnnData


def cluster_and_embed(
    adata: AnnData,
    *,
    rep_key: str = "X_pca",
    n_neighbors: int = 15,
    metric: str = "cosine",
    clustering_method: Literal["leiden", "louvain", "none"] = "leiden",
    resolution: float = 0.6,
    cluster_key: Optional[str] = None,
    umap_min_dist: float = 0.3,
    umap_spread: float = 1.0,
    random_state: int = 0,
) -> AnnData:
    """
    Build a KNN graph on a given embedding, optionally run clustering,
    and compute a UMAP embedding.

    Parameters
    ----------
    adata
        AnnData object with an embedding in `adata.obsm[rep_key]`.

    rep_key
        Name of the embedding to use for neighbor graph construction
        (e.g. "X_pca" or "X_pca_harmony").

    n_neighbors
        Number of neighbors for the KNN graph.

    metric
        Distance metric for the neighbor graph (e.g. "cosine" or "euclidean").

    clustering_method
        Which clustering algorithm to use:
            - "leiden"  (default, recommended)
            - "louvain"
            - "none"    → build neighbors + UMAP only, no cluster labels.

    resolution
        Resolution parameter passed to Leiden or Louvain (ignored if
        `clustering_method="none"`).

    cluster_key
        Name of the column in `adata.obs` where cluster labels will be stored.
        If None (default), uses:
            - "leiden"  when `clustering_method="leiden"`
            - "louvain" when `clustering_method="louvain"`

    umap_min_dist
        `min_dist` parameter for UMAP (controls how tightly points are packed).

    umap_spread
        `spread` parameter for UMAP (overall scale of the embedding).

    random_state
        Random seed for clustering and UMAP to ensure reproducibility.

    Returns
    -------
    AnnData
        The same AnnData object, updated with:
            - neighbor graph in `.uns["neighbors"]` / `.obsp["distances"]`
              and `.obsp["connectivities"]`
            - optional cluster labels in `adata.obs[cluster_key]`
            - UMAP coordinates in `.obsm["X_umap"]`
    """
    if rep_key not in adata.obsm:
        raise ValueError(
            f"'{rep_key}' not found in adata.obsm. "
            "Did you run PCA or batch correction?"
        )

    # ------------------------------------------------------------------
    # 1) Build neighbor graph from the chosen representation
    # ------------------------------------------------------------------
    sc.pp.neighbors(
        adata,
        use_rep=rep_key,
        n_neighbors=n_neighbors,
        metric=metric,
    )

    # ------------------------------------------------------------------
    # 2) Graph-based clustering (optional)
    # ------------------------------------------------------------------
    if clustering_method == "leiden":
        if cluster_key is None:
            cluster_key = "leiden"
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added=cluster_key,
            random_state=random_state,
        )

    elif clustering_method == "louvain":
        if cluster_key is None:
            cluster_key = "louvain"
        sc.tl.louvain(
            adata,
            resolution=resolution,
            key_added=cluster_key,
            random_state=random_state,
        )

    elif clustering_method == "none":
        # Skip clustering: users can run their own method later if they wish.
        cluster_key = None

    else:
        raise ValueError(
            f"Unknown clustering_method '{clustering_method}'. "
            "Use 'leiden', 'louvain', or 'none'."
        )

    # ------------------------------------------------------------------
    # 3) UMAP embedding
    # ------------------------------------------------------------------
    sc.tl.umap(
        adata,
        min_dist=umap_min_dist,
        spread=umap_spread,
        random_state=random_state,
    )

    return adata
