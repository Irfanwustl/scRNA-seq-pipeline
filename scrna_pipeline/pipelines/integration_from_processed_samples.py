"""
Strict integration pipeline for *already processed per-sample* scRNA-seq data.

This preset consumes a folder of per-sample `.h5ad` files that were produced
by a per-sample preprocessing pipeline (e.g. `standard_per_sample_pipeline`)
and performs **only cross-sample operations**:

    combine_samples → build_embedding_pca → batch_correction

Input contract (validated by failure, not branching logic)
----------------------------------------------------------
Each input `.h5ad` must:
- represent exactly one biological sample
- already be QCed, filtered, normalized, and doublet-removed
- contain required expression layers (e.g. `counts`, `log1p`)
- expose a stable sample identifier (e.g. `obs["sample_id"]`)

This pipeline intentionally does NOT:
- redo QC, Scrublet, or normalization
- perform clustering or annotation
- accept raw or partially processed data

Any violation of these assumptions raises immediately.

Use this preset to establish a **clean, reproducible boundary**
between per-sample processing and joint multi-sample analysis.
"""

from __future__ import annotations

from scrna_pipeline.core.pipeline import StepSpec

from scrna_pipeline.steps.concat_samples import CombineSamplesStep
from scrna_pipeline.steps.preprocessing import BuildEmbeddingPCAStep
from scrna_pipeline.steps.batch_correction import BatchCorrectionStep


def integration_from_processed_samples_pipeline(
    *,
    folder: str,
    pattern: str = "*.h5ad",

    # embedding PCA (recomputed on combined object)
    batch_key: str = "sample_id",
    hvg_flavor: str = "seurat_v3",
    n_top_genes: int = 2000,
    n_pcs: int = 50,

    # batch correction
    batch_method: str = "harmony",   # "none" or "harmony"
    rep_in: str = "X_pca",
) -> list[StepSpec]:
    """
    Build a strict integration pipeline from processed per-sample inputs.
    """

    return [
        StepSpec(
            name="combine_samples",
            step=CombineSamplesStep(
                folder=folder,
                pattern=pattern,
            ),
            mode="required",
            on_error="raise",
        ),

        StepSpec(
            name="build_embedding_pca",
            step=BuildEmbeddingPCAStep(
                batch_key=batch_key,
                hvg_flavor=hvg_flavor,
                n_top_genes=n_top_genes,
                n_pcs=n_pcs,
                params={"require_log1p_layer": True},
            ),
            mode="required",
            on_error="raise",
        ),

        StepSpec(
            name="batch_correction",
            step=BatchCorrectionStep(
                method=batch_method,
                batch_key=batch_key if batch_method != "none" else None,
                rep_in=rep_in,
            ),
            mode="required",
            on_error="raise",
        ),
    ]
