"""
Per-sample export workflow.

This module splits a (potentially multi-sample) AnnData (.h5ad) object into
one raw AnnData file per biological sample.

Responsibilities
----------------
- Resolve sample identity per cell (via obs column, resolver, or provided ID)
- Write one raw .h5ad per sample under `raw/per_sample/`
- Preserve all existing layers, embeddings, and metadata (no computation)

Non-responsibilities
--------------------
- No QC, normalization, HVG selection, or annotation
- No batch correction or integration
- No deletion of intermediate files

Typical usage
-------------
export_per_sample_h5ads(
    "raw/all_samples.h5ad",
    out_dir="raw/per_sample",
    sample_key="sample",
)
"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import re

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData


SampleResolver = Callable[[AnnData], pd.Series]


@dataclass(frozen=True)
class ExportReport:
    out_paths: List[Path]
    n_cells_per_sample: pd.Series


def _sanitize_filename(name: str, *, max_len: int = 200) -> str:
    """
    Make a string safe for filesystem paths across OSes.
    """
    name = str(name).strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name)
    name = name.strip("._-")
    if not name:
        name = "sample"
    return name[:max_len]


def resolve_sample_ids(
    adata: AnnData,
    *,
    sample_key: str | None = None,
    sample_id: str | None = None,
    sample_resolver: SampleResolver | None = None,
) -> pd.Series:
    """
    Resolve per-cell sample IDs for splitting.

    Priority:
      1) sample_id (treat entire AnnData as a single sample)
      2) sample_resolver(adata) -> pd.Series (index aligned to adata.obs_names)
      3) adata.obs[sample_key]

    Returns
    -------
    pd.Series
        Index = adata.obs_names, values = sample IDs (strings)
    """
    if sample_id is not None:
        s = pd.Series([str(sample_id)] * adata.n_obs, index=adata.obs_names, name="sample_id")
        return s

    if sample_resolver is not None:
        s = sample_resolver(adata)
        if not isinstance(s, pd.Series):
            raise TypeError("sample_resolver must return a pandas Series.")
        if len(s) != adata.n_obs:
            raise ValueError(
                f"sample_resolver returned length {len(s)} but adata has n_obs={adata.n_obs}."
            )
        # Ensure index alignment
        if not s.index.equals(adata.obs_names):
            try:
                s = s.reindex(adata.obs_names)
            except Exception as e:
                raise ValueError(
                    "sample_resolver Series index must match adata.obs_names (or be reindexable)."
                ) from e
        s = s.astype(str)
        s.name = "sample_id"
        return s

    if sample_key is None:
        raise ValueError("Provide one of: sample_key, sample_id, or sample_resolver.")

    if sample_key not in adata.obs.columns:
        raise KeyError(
            f"sample_key='{sample_key}' not found in adata.obs. "
            "Provide sample_id for single-sample input, or sample_resolver for custom extraction."
        )

    s = adata.obs[sample_key].astype(str).copy()
    s.index = adata.obs_names
    s.name = "sample_id"
    return s


def export_per_sample_h5ads(
    multi_h5ad_path: str | Path,
    out_dir: str | Path,
    *,
    sample_key: str | None = None,
    sample_id: str | None = None,
    sample_resolver: SampleResolver | None = None,
    # Optional standardization for downstream stages:
    set_sample_id_col: str = "sample_id",
    set_batch_id_col: str = "batch",
    batch_key: str | None = None,
    # Safety / hygiene:
    make_obs_names_unique: bool = True,
    obs_name_prefix_sep: str = "__",
    # What to keep:
    keep_uns: bool = True,
    keep_obsm: bool = True,
    keep_obsp: bool = True,
    keep_varm: bool = True,
    # Output:
    filename_suffix: str = "",
    compression: str | None = "gzip",
    overwrite: bool = False,
    dry_run: bool = False,
) -> ExportReport:
    """
    Split a (potentially) multi-sample .h5ad into per-sample .h5ad files.

    This is designed as an IO boundary:
      - Writes per-sample files to disk
      - Returns the written paths (and a small report)

    Parameters
    ----------
    multi_h5ad_path
        Path to input .h5ad containing one or many samples.

    out_dir
        Directory to write per-sample .h5ad files.

    sample_key
        Column in adata.obs that identifies sample per cell.

    sample_id
        If provided, treat the entire input as a single sample with this ID.

    sample_resolver
        Callable that returns a pd.Series of sample IDs per cell (index aligned to adata.obs_names).

    set_sample_id_col
        Column name to create/overwrite in each per-sample adata.obs to store the resolved sample id.

    set_batch_id_col, batch_key
        If batch_key provided and exists in obs, set batch column from it.
        Else batch defaults to sample_id.

    make_obs_names_unique
        If True, prefixes cell barcodes with sample id to prevent collisions after concat.

    keep_uns/obsm/obsp/varm
        If False, drop those containers in output to reduce file size.

    filename_suffix
        Optional suffix added to each output filename before ".h5ad".

    compression
        Compression for writing .h5ad. "gzip" is common. Set None for no compression.

    overwrite
        If False, will error if an output file already exists.

    dry_run
        If True, do not write any files; just return planned paths/report.

    Returns
    -------
    ExportReport
        out_paths: list of written file paths
        n_cells_per_sample: Series mapping sample_id -> number of cells
    """
    multi_h5ad_path = Path(multi_h5ad_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(multi_h5ad_path)

    sample_ids = resolve_sample_ids(
        adata,
        sample_key=sample_key,
        sample_id=sample_id,
        sample_resolver=sample_resolver,
    )

    # Decide batch IDs (standardized column inside each per-sample file)
    if batch_key is not None:
        if batch_key not in adata.obs.columns:
            raise KeyError(f"batch_key='{batch_key}' not found in adata.obs.")
        batch_ids = adata.obs[batch_key].astype(str)
        batch_ids.index = adata.obs_names
    else:
        batch_ids = sample_ids  # default

    # Compute counts for report and ensure deterministic order
    counts = sample_ids.value_counts().sort_index()
    unique_samples = list(counts.index)

    # Filename suffix normalization
    suffix = filename_suffix.strip()
    if suffix and not suffix.startswith("."):
        suffix = "." + suffix

    out_paths: List[Path] = []

    for sid in unique_samples:
        safe_sid = _sanitize_filename(sid)
        out_path = out_dir / f"{safe_sid}{suffix}.h5ad"

        if out_path.exists() and not overwrite and not dry_run:
            raise FileExistsError(
                f"Output already exists: {out_path}. Set overwrite=True to replace."
            )

        out_paths.append(out_path)

        if dry_run:
            continue

        mask = (sample_ids == sid).to_numpy()
        if not np.any(mask):
            continue

        sub = adata[mask].copy()

        # Standardize obs columns
        sub.obs[set_sample_id_col] = str(sid)
        # batch per cell (aligned subset)
        sub.obs[set_batch_id_col] = batch_ids.loc[sub.obs_names].astype(str).values

        # Make obs_names unique across samples
        if make_obs_names_unique:
            # Prefix old names; ensure uniqueness
            pref = f"{safe_sid}{obs_name_prefix_sep}"
            new_names = [pref + str(x) for x in sub.obs_names]
            sub.obs_names = pd.Index(new_names)
            sub.obs_names_make_unique()

        # Optional drops to reduce size
        if not keep_uns:
            sub.uns = {}
        if not keep_obsm:
            sub.obsm = {}
        if not keep_obsp:
            sub.obsp = {}
        if not keep_varm:
            sub.varm = {}

        # Write
        if compression is None:
            sub.write_h5ad(out_path)
        else:
            sub.write_h5ad(out_path, compression=compression)

    return ExportReport(out_paths=out_paths, n_cells_per_sample=counts)
