"""
Batch correction helpers.

This module contains small, focused functions that take an AnnData with
`adata.obsm["X_pca"]` and optionally:

- do nothing (no batch correction)
- run Harmony to integrate batches

The dispatcher `apply_batch_correction` chooses the method and returns the
name of the embedding (obsm key) that should be used for neighbors/UMAP.
"""

from __future__ import annotations

from typing import Callable, Dict

import scanpy.external as sce
from anndata import AnnData


def batch_none(adata: AnnData, *, rep_in: str = "X_pca") -> str:
    """
    No batch correction. Just checks that the PCA embedding exists and
    returns the name to use.

    Parameters
    ----------
    adata
        AnnData with PCA in adata.obsm[rep_in].

    rep_in
        Name of the PCA embedding to use (default "X_pca").

    Returns
    -------
    str
        The same representation name (rep_in).
    """
    if rep_in not in adata.obsm:
        raise ValueError(f"'{rep_in}' not found in adata.obsm. Did you run PCA?")
    return rep_in


def batch_harmony(
    adata: AnnData,
    *,
    batch_key: str,
    rep_in: str = "X_pca",
    rep_out: str = "X_pca_harmony",
    verbose: bool = True,
) -> str:
    """
    Run Harmony batch correction on PCA coordinates.

    This uses Scanpy's wrapper around Harmony. It reads `adata.obsm[rep_in]`
    (usually "X_pca") and writes the Harmony-corrected embedding to
    `adata.obsm["X_pca_harmony"]` by default.

    Parameters
    ----------
    adata
        AnnData with PCA already computed.
    batch_key
        Column in adata.obs that encodes the batch / sample.
    rep_in
        Input representation to correct (default: "X_pca").
    rep_out
        Name of the corrected representation (default: "X_pca_harmony").

    Returns
    -------
    str
        Name of the representation that should be used downstream (rep_out).
    """
    if rep_in not in adata.obsm:
        raise ValueError(f"'{rep_in}' not found in adata.obsm. Did you run PCA?")

    if verbose:
        print(f"Running Harmony batch correction using batch_key='{batch_key}'...")

    # Scanpy's Harmony wrapper modifies adata in-place and creates "X_pca_harmony"
    sce.pp.harmony_integrate(adata, key=batch_key)

    # Harmony uses a fixed output name "X_pca_harmony", so rep_out is mostly
    # here for completeness if you want to follow the convention.
    if rep_out not in adata.obsm:
        # If Scanpy/Harmony changes behavior in future, this is a safeguard.
        raise ValueError(
            f"Expected Harmony output '{rep_out}' in adata.obsm, but it was not found."
        )

    return rep_out


# Mapping from method name to implementation function
BATCH_METHODS: Dict[str, Callable[..., str]] = {
    "none": batch_none,
    "harmony": batch_harmony,
    # In future you can easily add:
    # "bbknn": batch_bbknn,
    # "scvi": batch_scvi,
}


def apply_batch_correction(
    adata: AnnData,
    *,
    method: str = "none",
    batch_key: str | None = None,
    rep_in: str = "X_pca",
    verbose: bool = True,
    **kwargs,
) -> str:
    """
    Apply a batch correction method and return the name of the embedding
    to use for downstream steps (neighbors, clustering, UMAP).

    Parameters
    ----------
    adata
        AnnData object with PCA embedding in `adata.obsm[rep_in]`.

    method
        Batch correction method name. Currently supported:
        - "none"    : no batch correction
        - "harmony" : Harmony integration on PCA

    batch_key
        Column in `adata.obs` with batch labels. Required for methods that
        actually perform batch correction (e.g. Harmony).

    rep_in
        Name of the embedding to correct (default: "X_pca").

    verbose
        If True, print messages about what is being run.

    kwargs
        Any extra keyword arguments passed to the specific batch function.

    Returns
    -------
    str
        Name of the embedding that should be used downstream.
    """
    if method not in BATCH_METHODS:
        raise ValueError(f"Unknown batch correction method: '{method}'")

    # Methods other than "none" generally require a batch_key.
    if method != "none" and batch_key is None:
        raise ValueError(
            f"Batch method '{method}' requires a non-None batch_key."
        )

    func = BATCH_METHODS[method]

    if method == "none":
        # No batch_key needed
        return func(adata, rep_in=rep_in, **kwargs)
    elif method == "harmony":
        # Pass batch_key explicitly
        return func(
            adata,
            batch_key=batch_key,  # type: ignore[arg-type]
            rep_in=rep_in,
            verbose=verbose,
            **kwargs,
        )
