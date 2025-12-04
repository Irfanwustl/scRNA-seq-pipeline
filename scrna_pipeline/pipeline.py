from __future__ import annotations
from typing import Any, Dict

import pprint

import scanpy as sc
from anndata import AnnData

from .preprocessing import preprocess_to_pca
from .batch_correction import apply_batch_correction
from .clustering import cluster_and_embed


def _short_dict(d: Dict[str, Any], max_items: int = 5) -> str:
    """
    Nicely format a dict for display, showing at most `max_items` entries.

    If the dict is longer, we show only the first few items and append:
        ... (+N more)

    This avoids flooding the console when users pass large kwargs.
    """
    import pprint

    if not d:
        return "{}"

    items = list(d.items())

    # If few enough items, just pretty-print the whole dict
    if len(items) <= max_items:
        return pprint.pformat(d, compact=True)

    # Otherwise, show only the first `max_items`
    head = dict(items[:max_items])
    formatted_head = pprint.pformat(head, compact=True)

    # Construct suffix like: ... (+3 more)
    suffix = f"... (+{len(items) - max_items} more)"

    # Insert the suffix before the closing brace
    # Example: "{'a': 1, 'b': 2, ... (+5 more)}"
    return formatted_head[:-1] + ", " + suffix + "}"



def standard_scrna_pipeline(
    adata: AnnData,
    *,
    batch_key: str | None = "sample",
    batch_method: str = "harmony",   # "none" or "harmony"
    hvg_flavor: str = "seurat_v3",
    n_top_genes: int = 2000,
    n_pcs: int = 50,
    n_neighbors: int = 15,
    # generic, not tied to louvain
    clustering_method: str = "leiden",   # "leiden", "louvain", "none"
    cluster_resolution: float = 0.6,
    preprocess_kwargs: Dict[str, Any] | None = None,
    batch_kwargs: Dict[str, Any] | None = None,
    cluster_kwargs: Dict[str, Any] | None = None,
    verbose: bool = True,
) -> AnnData:
    """
    Execute a complete, end-to-end scRNA-seq analysis workflow.

    This function provides a high-level, user-friendly pipeline that performs:
      1) Preprocessing + QC + HVGs + PCA             → `preprocess_to_pca`
      2) Optional batch correction / integration     → `apply_batch_correction`
      3) KNN graph + clustering + UMAP embedding    → `cluster_and_embed`

    The clustering step is configurable:
        - Default: Leiden (`clustering_method="leiden"`)
        - Optional: Louvain (`clustering_method="louvain"`)
        - Or: skip clustering (`clustering_method="none"`)

    All additional keyword arguments for each sub-step can be passed via:
        - `preprocess_kwargs`
        - `batch_kwargs`
        - `cluster_kwargs`

    If `verbose=True`, a concise summary of the pipeline configuration and
    key steps is printed. The same configuration is stored in:

        adata.uns["scrna_pipeline"]["standard_pipeline"]

    for reproducibility.
    """

    # Ensure keyword-argument dictionaries exist
    if preprocess_kwargs is None:
        preprocess_kwargs = {}
    if batch_kwargs is None:
        batch_kwargs = {}
    if cluster_kwargs is None:
        cluster_kwargs = {}

    if verbose:
        print("\n[ scrna-pipeline ] Running standard scRNA-seq pipeline")
        print("  • Input AnnData shape: "
              f"{adata.n_obs} cells × {adata.n_vars} genes")
        print("  • Batch key:        ", repr(batch_key))
        print("  • Batch method:     ", repr(batch_method))
        print("  • HVG flavor:       ", repr(hvg_flavor))
        print("  • n_top_genes:      ", n_top_genes)
        print("  • n_pcs:            ", n_pcs)
        print("  • Clustering:       ", repr(clustering_method))
        print("  • Cluster resol.:   ", cluster_resolution)
        print("  • Preprocess kwargs:", _short_dict(preprocess_kwargs))
        print("  • Batch kwargs:     ", _short_dict(batch_kwargs))
        print("  • Cluster kwargs:   ", _short_dict(cluster_kwargs))
        print("")

    # ----------------------------------------------------------------------
    # (1) Preprocess → PCA
    # ----------------------------------------------------------------------
    adata = preprocess_to_pca(
        adata,
        batch_key=batch_key,
        hvg_flavor=hvg_flavor,
        n_top_genes=n_top_genes,
        n_pcs=n_pcs,
        **preprocess_kwargs,
    )

    if verbose:
        print("[ scrna-pipeline ] Step 1 complete: preprocess_to_pca")
        print("  • PCA stored in .obsm['X_pca']")
        print("  • Genes after HVG selection:", adata.n_vars)
        print("  • Cells after QC / Scrublet:", adata.n_obs)
        print("")

    # ----------------------------------------------------------------------
    # (2) Batch correction / integration
    # ----------------------------------------------------------------------
    rep_key = apply_batch_correction(
        adata,
        method=batch_method,
        batch_key=batch_key,
        rep_in="X_pca",
        **batch_kwargs,
    )

    if verbose:
        if batch_method == "none":
            print("[ scrna-pipeline ] Step 2: no batch correction (using 'X_pca').")
        else:
            print(f"[ scrna-pipeline ] Step 2 complete: batch correction with '{batch_method}'")
            print(f"  • Corrected embedding stored in .obsm['{rep_key}']")
        print("")

    # ----------------------------------------------------------------------
    # (3) Neighborhood graph → clustering → UMAP embedding
    # ----------------------------------------------------------------------
    adata = cluster_and_embed(
        adata,
        rep_key=rep_key,
        n_neighbors=n_neighbors,
        clustering_method=clustering_method,
        resolution=cluster_resolution,
        **cluster_kwargs,
    )

    if verbose:
        print("[ scrna-pipeline ] Step 3 complete: clustering + UMAP")
        if clustering_method != "none":
            # cluster_and_embed will have chosen "leiden" / "louvain" or custom cluster_key
            # but we don't know the exact key_name here without inspecting adata.obs.
            # We can guess common keys:
            for key in ("leiden", "louvain", "cluster"):
                if key in adata.obs:
                    print(f"  • Cluster labels in .obs['{key}']")
                    break
        print("  • UMAP stored in .obsm['X_umap']")
        print("  • Final AnnData shape: "
              f"{adata.n_obs} cells × {adata.n_vars} genes\n")

    # ----------------------------------------------------------------------
    # Store configuration in .uns for reproducibility
    # ----------------------------------------------------------------------
    cfg = {
        "batch_key": batch_key,
        "batch_method": batch_method,
        "hvg_flavor": hvg_flavor,
        "n_top_genes": n_top_genes,
        "n_pcs": n_pcs,
        "n_neighbors": n_neighbors,
        "clustering_method": clustering_method,
        "cluster_resolution": cluster_resolution,
        "preprocess_kwargs": preprocess_kwargs,
        "batch_kwargs": batch_kwargs,
        "cluster_kwargs": cluster_kwargs,
    }
    if "scrna_pipeline" not in adata.uns:
        adata.uns["scrna_pipeline"] = {}
    adata.uns["scrna_pipeline"]["standard_pipeline"] = cfg

    return adata
