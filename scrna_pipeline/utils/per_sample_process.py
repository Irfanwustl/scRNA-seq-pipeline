"""
Per-sample processing EXECUTOR.

Design goals
-----------
1) Keep the executor stable ("boring"):
   - No hardcoded biological logic (no preprocess/clustering/annotation decisions).
   - The pipeline is defined by user-supplied `steps: list[StepSpec]`.

2) Separate responsibilities cleanly:
   - `process_sample(...)` does ONLY processing and returns AnnData (no saving).
   - `process_all_samples(...)` handles I/O + save policy (drop graphs) + writing.

Why this matters
----------------
- Interactive debugging: you can call `process_sample(...)` in a notebook and inspect UMAP etc.
- Batch runs: `process_all_samples(...)` can drop graphs before saving to keep disk usage small.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import scanpy as sc
from anndata import AnnData

from scrna_pipeline.core.step import StepContext, StepSpec, run_steps


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _ensure_obs_cols(adata: AnnData, cols: Sequence[str]) -> None:
    """Fail fast if required obs columns are missing."""
    missing = [c for c in cols if c not in adata.obs.columns]
    if missing:
        raise KeyError(f"Missing required obs columns: {missing}")


def _drop_graphs_inplace(adata: AnnData) -> None:
    """
    Drop large graph objects to reduce .h5ad size.

    Clears:
      - adata.obsp (distances/connectivities, etc.)

    Keeps:
      - embeddings like adata.obsm["X_pca"], adata.obsm["X_umap"] (cheap and useful)
      - annotations in adata.obs
      - layers like counts/log1p
    """
    adata.obsp.clear()
    # Optional extra shrinkage (use only if you are sure you don't need it):
    # adata.uns.pop("neighbors", None)


def _iter_input_files(in_dir: Path, pattern: str) -> Iterable[Path]:
    """Yield input files in deterministic order."""
    yield from sorted(in_dir.glob(pattern))


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def process_sample(
    adata: AnnData,
    *,
    steps: list[StepSpec],
    required_obs: list[str] | None = None,
    ctx: StepContext | None = None,
) -> AnnData:
    """
    Process ONE sample AnnData in-memory and return the processed AnnData.

    This function performs NO I/O (no reading, no writing).
    It is intended to be used both:
      - interactively (notebook): inspect result directly
      - in batch drivers: process -> apply save policy -> write to disk

    Parameters
    ----------
    adata
        Per-sample AnnData (already loaded).

    steps
        User-supplied list of StepSpec objects (your pipeline definition).

    required_obs
        Minimal obs columns required by your conventions.
        Default: ["sample_id", "batch"].

    ctx
        Execution context (verbosity, strictness, etc.). If None, a default is created.

    Returns
    -------
    AnnData
        Processed AnnData (new object if steps create copies; otherwise same object mutated).
    """
    if required_obs is None:
        required_obs = ["sample_id", "batch"]

    _ensure_obs_cols(adata, required_obs)

    if ctx is None:
        ctx = StepContext(verbose=True, strict=True)

    return run_steps(adata, steps, ctx=ctx)


def process_all_samples(
    raw_per_sample_dir: str | Path,
    processed_dir: str | Path,
    *,
    steps: list[StepSpec],
    pattern: str = "*.h5ad",
    required_obs: list[str] | None = None,
    ctx: StepContext | None = None,
    # Batch save policy
    drop_graphs_before_save: bool = True,
    compression: str = "gzip",
    output_suffix: str = ".processed.h5ad",
) -> list[Path]:
    """
    Batch driver: read each per-sample .h5ad, process it, then save it.

    Responsibilities
    ----------------
    - I/O (read inputs, write outputs)
    - Save policy (drop graphs before saving)
    - Deterministic iteration order

    Parameters
    ----------
    raw_per_sample_dir
        Directory containing raw per-sample .h5ad files.

    processed_dir
        Output directory where processed files are written.

    steps
        Pipeline definition (StepSpec list) applied to all samples.

    pattern
        Glob pattern to select input files (default "*.h5ad").

    required_obs, ctx
        Passed through to `process_sample()`.

    drop_graphs_before_save
        If True (default), clears `.obsp` before writing outputs to reduce file size.

    compression
        Compression used by AnnData writer.

    output_suffix
        Suffix appended to each input stem for output naming.

    Returns
    -------
    list[Path]
        Output paths, aligned with sorted input order.
    """
    raw_per_sample_dir = Path(raw_per_sample_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if ctx is None:
        ctx = StepContext(verbose=True, strict=True)

    out_paths: list[Path] = []

    for in_path in _iter_input_files(raw_per_sample_dir, pattern):
        adata = sc.read_h5ad(in_path)

        # processing only (no saving inside)
        adata = process_sample(
            adata,
            steps=steps,
            required_obs=required_obs,
            ctx=ctx,
        )

        # batch-only save policy
        if drop_graphs_before_save:
            _drop_graphs_inplace(adata)
            adata.uns.setdefault("scrna_pipeline", {})
            adata.uns["scrna_pipeline"]["dropped_graphs_before_save"] = True

        out_path = processed_dir / f"{in_path.stem}{output_suffix}"
        adata.write_h5ad(out_path, compression=compression)
        out_paths.append(out_path)

    return out_paths
