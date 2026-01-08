"""
Batch execution utilities for scRNA-seq pipelines.

This module provides a generic runner for executing *per-sample pipelines*
over multiple AnnData objects and persisting the results to disk.

Design goals
------------
- Pipeline-agnostic:
  The runner does not know or care which steps are executed. It only requires
  a pipeline factory that returns a list of StepSpec objects.

- Separation of concerns:
  * Pipeline presets* define *what* to run (step composition).
  * This runner* defines *how* to run pipelines across many samples
  (iteration, execution context, saving, graph stripping).

- In-memory first:
  This runner operates on AnnData objects already loaded in memory.
  It is intentionally not a workflow engine (Snakemake/Nextflow).
  It can later be wrapped by one.

Key abstractions
----------------
- PipelineFactory:
    Callable that builds a list of StepSpec objects.
- KwargsFactory:
    Callable that maps (sample_name, AnnData) -> pipeline keyword arguments.
    This allows both constant and per-sample configuration.
- BatchRunConfig:
    Centralized configuration for execution behavior and output persistence.

Typical usage
-------------
    cfg = BatchRunConfig(out_dir=Path("results/per_sample"))

    kwargs_factory = constant_kwargs_factory({
        "batch_key": "sample",
        "marker_dict": marker_dict,
    })

    run_pipeline_on_batch(
        adatas,
        pipeline=standard_per_sample_pipeline,
        kwargs_factory=kwargs_factory,
        config=cfg,
    )

Notes
-----
- Neighbor graphs (adata.obsp / adata.uns["neighbors"]) can optionally be
  stripped before saving to reduce file size.
- This module is intentionally small and opinionated; advanced scheduling,
  caching, and resource management should be handled by an external workflow
  engine if needed.
"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple

import scanpy as sc
from anndata import AnnData

from scrna_pipeline.core.pipeline import StepSpec, StepContext, run_steps


# ------------------------------------------------------------
# Types
# ------------------------------------------------------------
PipelineFactory = Callable[..., list[StepSpec]]
KwargsFactory = Callable[[str, AnnData], dict[str, Any]]
SampleCallback = Callable[[str, AnnData], None]


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
@dataclass(frozen=True)
class BatchRunConfig:
    out_dir: Path
    filename_suffix: str = ".processed.h5ad"
    compression: str = "gzip"

    # execution
    verbose: bool = True
    strict: bool = True

    # saving
    strip_graph: bool = True
    keep_umap: bool = True
    keep_pca: bool = True


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def constant_kwargs_factory(kwargs: dict[str, Any]) -> KwargsFactory:
    """
    Return a kwargs factory that yields the same kwargs for every sample.

    A shallow copy is returned each time to avoid accidental shared mutation.
    """
    def _factory(_name: str, _adata: AnnData) -> dict[str, Any]:
        return dict(kwargs)

    return _factory


def strip_graph_for_disk(
    adata: AnnData,
    *,
    keep_umap: bool = True,
    keep_pca: bool = True,
) -> AnnData:
    """
    Return a COPY of adata with neighbor graphs removed to reduce file size.
    Keeps X_umap/X_pca by default.
    """
    a = adata.copy()

    if not keep_umap:
        a.obsm.pop("X_umap", None)
    if not keep_pca:
        a.obsm.pop("X_pca", None)

    # scanpy neighbor graph artifacts
    a.obsp.pop("connectivities", None)
    a.obsp.pop("distances", None)
    a.uns.pop("neighbors", None)

    return a


def _iter_named_adatas(
    adatas: Mapping[str, AnnData] | Iterable[Tuple[str, AnnData]],
) -> Iterable[Tuple[str, AnnData]]:
    return adatas.items() if hasattr(adatas, "items") else adatas


# ------------------------------------------------------------
# Single-sample core (shared)
# ------------------------------------------------------------
def _run_one_sample(
    name: str,
    adata: AnnData,
    *,
    pipeline: PipelineFactory,
    kwargs_factory: KwargsFactory,
    config: BatchRunConfig,
    ctx: StepContext,
    on_sample_done: Optional[SampleCallback] = None,
) -> Path:
    """
    Run the pipeline for a single sample and write its output.

    Returns the saved output path.
    """
    if config.verbose:
        pname = getattr(pipeline, "__name__", pipeline.__class__.__name__)
        print(f"\n=== [{name}] running {pname} ===")

    kwargs = kwargs_factory(name, adata)
    steps = pipeline(**kwargs)

    adata_proc = run_steps(adata, steps, ctx=ctx)

    if on_sample_done is not None:
        on_sample_done(name, adata_proc)

    adata_to_write = (
        strip_graph_for_disk(
            adata_proc,
            keep_umap=config.keep_umap,
            keep_pca=config.keep_pca,
        )
        if config.strip_graph
        else adata_proc
    )

    out_path = config.out_dir / f"{name}{config.filename_suffix}"
    adata_to_write.write_h5ad(out_path, compression=config.compression)

    if config.verbose:
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"=== [{name}] saved: {out_path} ({size_mb:.1f} MB) ===")

    return out_path


# ------------------------------------------------------------
# Batch runners
# ------------------------------------------------------------
def run_pipeline_on_batch(
    adatas: Mapping[str, AnnData] | Iterable[Tuple[str, AnnData]],
    *,
    pipeline: PipelineFactory,
    kwargs_factory: KwargsFactory,
    config: BatchRunConfig,
    on_sample_done: Optional[SampleCallback] = None,
) -> dict[str, Path]:
    """
    Run a per-sample pipeline for each in-memory AnnData and save outputs.
    """
    config.out_dir.mkdir(parents=True, exist_ok=True)
    ctx = StepContext(verbose=config.verbose, strict=config.strict)

    saved: dict[str, Path] = {}
    for name, adata in _iter_named_adatas(adatas):
        out_path = _run_one_sample(
            name,
            adata,
            pipeline=pipeline,
            kwargs_factory=kwargs_factory,
            config=config,
            ctx=ctx,
            on_sample_done=on_sample_done,
        )
        saved[name] = out_path

    return saved


def run_pipeline_on_h5ad_folder(
    in_dir: str | Path,
    *,
    pipeline: PipelineFactory,
    kwargs_factory: KwargsFactory,
    config: BatchRunConfig,
    pattern: str = "*.h5ad",
    delete_inputs: bool = False,
    on_sample_done: Optional[SampleCallback] = None,
) -> dict[str, Path]:
    """
    Run a per-sample pipeline for each .h5ad in a folder and save outputs.

    Notes
    -----
    - Streams one file at a time (does not keep all adatas in memory).
    - If delete_inputs=True, deletes each input file only after its output
      is successfully written.
    """
    in_dir = Path(in_dir)
    paths = sorted(in_dir.glob(pattern))
    if len(paths) == 0:
        raise FileNotFoundError(f"No files matching {pattern} under {in_dir}")

    config.out_dir.mkdir(parents=True, exist_ok=True)
    ctx = StepContext(verbose=config.verbose, strict=config.strict)

    saved: dict[str, Path] = {}
    for p in paths:
        name = p.stem

        if config.verbose:
            print(f"\n=== [{name}] loading: {p} ===")

        adata = sc.read_h5ad(p)

        out_path = _run_one_sample(
            name,
            adata,
            pipeline=pipeline,
            kwargs_factory=kwargs_factory,
            config=config,
            ctx=ctx,
            on_sample_done=on_sample_done,
        )
        saved[name] = out_path

        if delete_inputs:
            p.unlink()
            if config.verbose:
                print(f"=== [{name}] deleted input: {p} ===")

    return saved
