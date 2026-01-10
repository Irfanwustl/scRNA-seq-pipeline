"""
Integration runner utilities.

This module provides a small, opinionated runner for **integration-stage** pipelines
(i.e., pipelines that produce ONE combined AnnData from many per-sample inputs).

Why a separate runner?
----------------------
Your per-sample runner (`run_pipeline_on_h5ad_folder`) is optimized for:
    many inputs -> many outputs (one output per sample)

Integration pipelines are different:
    many inputs -> ONE output (a combined/integrated AnnData)

This runner keeps that boundary clean:
- Presets define *what* to run (StepSpec composition).
- This runner defines *how* to run it once, optionally save, and optionally
  attach post-run artifacts (e.g., UMAP pre/post) without polluting the preset.

Core features
-------------
- Runs any integration pipeline factory: `pipeline(**pipeline_kwargs) -> list[StepSpec]`
- Returns the integrated `AnnData` in memory
- Saving is OPTIONAL via `IntegrationRunConfig.out_path`
- Optional `postprocess(adata)` hook runs after the pipeline and before saving
  (ideal for visualization artifacts / assertions / metadata tweaks)

Design notes
------------
- Integration presets typically start with a "loader/combine" step (e.g. CombineSamplesStep)
  that constructs the real AnnData. Therefore, the runner seeds execution with a minimal
  dummy AnnData.

Typical examples
----------------

(1) Run integration in-memory only (no saving)
>>> adata_int = run_integration_pipeline(
...     pipeline=integration_from_processed_samples_pipeline,
...     pipeline_kwargs=dict(
...         folder="per_sample_processed",
...         batch_key="sample_id",
...         batch_method="harmony",
...     ),
...     config=IntegrationRunConfig(out_path=None),
... )

(2) Run integration and save one combined file
>>> from pathlib import Path
>>> adata_int = run_integration_pipeline(
...     pipeline=integration_from_processed_samples_pipeline,
...     pipeline_kwargs=dict(
...         folder="per_sample_processed",
...         batch_key="sample_id",
...         batch_method="harmony",
...     ),
...     config=IntegrationRunConfig(out_path=Path("results/combined.integrated.h5ad")),
... )

(3) Run integration, add UMAP before/after batch correction, then save
>>> def add_pre_post_umaps(a: AnnData) -> None:
...     # do something
>>>
>>> adata_int = run_integration_pipeline(
...     pipeline=integration_from_processed_samples_pipeline,
...     pipeline_kwargs=dict(
...         folder="per_sample_processed",
...         batch_key="sample_id",
...         batch_method="harmony",
...     ),
...     postprocess=add_pre_post_umaps,
...     config=IntegrationRunConfig(out_path=Path("results/combined.integrated.with_umaps.h5ad")),
... )
(4) Add assertions/sanity checks via postprocess (no saving)
>>> def assert_has_harmony(a: AnnData) -> None:
...     if "X_pca_harmony" not in a.obsm:
...         raise RuntimeError("Expected Harmony embedding but it is missing.")
>>>
>>> adata_int = run_integration_pipeline(
...     pipeline=integration_from_processed_samples_pipeline,
...     pipeline_kwargs=dict(folder="per_sample_processed/", batch_key="sample_id", batch_method="harmony"),
...     postprocess=assert_has_harmony,
...     config=IntegrationRunConfig(out_path=None),
... )

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import anndata as ad
from anndata import AnnData

from scrna_pipeline.core.pipeline import StepContext, StepSpec, run_steps


PipelineFactory = Callable[..., list[StepSpec]]


@dataclass(frozen=True)
class IntegrationRunConfig:
    """
    Configuration for a single integration pipeline run.

    Parameters
    ----------
    out_path
        If provided, save the integrated AnnData to this path.
        If None, do not save (in-memory only).

    compression
        Passed to `AnnData.write_h5ad` when saving. Common values:
        - "gzip" (default)
        - "lzf" (faster, often larger)

    verbose
        If True, runner prints a short save message and StepContext is verbose.

    strict
        If True, StepContext is strict (fail fast).
    """
    out_path: Optional[Path] = None
    compression: str = "gzip"
    verbose: bool = True
    strict: bool = True


def run_integration_pipeline(
    *,
    pipeline: PipelineFactory,
    pipeline_kwargs: dict[str, Any],
    config: IntegrationRunConfig,
    postprocess: Optional[Callable[[AnnData], None]] = None,
) -> AnnData:
    """
    Run an integration pipeline (single execution) and optionally save the result.

    This runner is intended for **integration-stage** pipelines that produce one
    combined AnnData from many per-sample inputs (often loaded/combined inside the
    first step, e.g. CombineSamplesStep).

    Parameters
    ----------
    pipeline
        A pipeline factory: `pipeline(**pipeline_kwargs) -> list[StepSpec]`.

    pipeline_kwargs
        Keyword arguments forwarded into the pipeline factory.

    config
        Execution + optional saving configuration.

    postprocess
        Optional callback invoked after the pipeline finishes and before saving.
        Use this for:
        - computing "UMAP pre/post" from X_pca vs X_pca_harmony
        - extra assertions
        - adding lightweight provenance metadata

        IMPORTANT:
        - Prefer side-effect-only edits to the provided AnnData.
        - If you want a more formal/typed postprocessing step, implement a Step.

    Returns
    -------
    AnnData
        The integrated AnnData (in memory). If config.out_path is provided, it is
        also written to disk.
    """
    ctx = StepContext(verbose=config.verbose, strict=config.strict)

    # Build the steps once
    steps = pipeline(**pipeline_kwargs)

    # Seed with a minimal placeholder AnnData. Integration pipelines generally
    # construct the real AnnData inside the first steps (e.g. combine/load).
    adata0 = ad.AnnData()

    # Execute
    adata_out = run_steps(adata0, steps, ctx=ctx)

    # Optional post-run hook
    if postprocess is not None:
        postprocess(adata_out)

    # Optional save
    if config.out_path is not None:
        out_path = Path(config.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        adata_out.write_h5ad(out_path, compression=config.compression)

        if config.verbose:
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"Saved integrated AnnData: {out_path} ({size_mb:.1f} MB)")

    return adata_out
