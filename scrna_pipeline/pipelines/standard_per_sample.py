"""
Strict per-sample pipeline preset.

This preset defines a fixed, linear per-sample workflow:
    preprocess → cluster+umap → broad annotation

Strict by default:
- Core steps are always executed (no implicit skipping)
- Missing inputs raise immediately

Optional steps (explicit opt-in):
- inferCNV can be appended when `run_infercnv=True` (requires reference categories)
"""


from __future__ import annotations

from typing import Dict, List

from scrna_pipeline.core.pipeline import StepSpec

from scrna_pipeline.steps.preprocessing import PreprocessToPCAStep
from scrna_pipeline.steps.clustering import ClusterAndEmbedStep
from scrna_pipeline.steps.annotation import BroadAnnotationStep



def standard_per_sample_pipeline(
    *,
    batch_key: str,
    marker_dict: Dict[str, List[str]],
    # preprocess
    hvg_flavor: str = "seurat",
    n_top_genes: int = 2000,
    n_pcs: int = 50,
    # clustering
    rep_key: str = "X_pca",
    n_neighbors: int = 15,
    clustering_method: str = "leiden",
    resolution: float = 0.6,
    cluster_key: str = "leiden",
    # annotation
    broad_label_key: str = "broad_celltype",

    # inferCNV (optional)
    run_infercnv: bool = False,
    infercnv_reference_categories: list[str] | None = None,
    infercnv_layer: str | None = None,
) -> list[StepSpec]:
    """
    Build a strict per-sample pipeline.

    Assumptions (validated by failure, not logic):
    - batch_key exists in adata.obs
    - clustering_method creates `cluster_key`
    - marker_dict is valid
    """

    steps: list[StepSpec] = [
        StepSpec(
            name="preprocess_to_pca",
            step=PreprocessToPCAStep(
                batch_key=batch_key,
                hvg_flavor=hvg_flavor,
                n_top_genes=n_top_genes,
                n_pcs=n_pcs,
            ),
            mode="required",
            on_error="raise",
        ),

        StepSpec(
            name="cluster_and_embed",
            step=ClusterAndEmbedStep(
                rep_key=rep_key,
                n_neighbors=n_neighbors,
                clustering_method=clustering_method,
                resolution=resolution,
                cluster_key=cluster_key,
            ),
            mode="required",
            on_error="raise",
        ),

        StepSpec(
            name="broad_annotation",
            step=BroadAnnotationStep(
                marker_dict=marker_dict,
                cluster_key=cluster_key,
                broad_label_key=broad_label_key,
                score_prefix="broad_",
                use_raw=None,
            ),
            mode="required",
            on_error="raise",
        ),

    ]


    if run_infercnv:
        if not infercnv_reference_categories:
            raise ValueError(
                "run_infercnv=True requires infercnv_reference_categories "
                f"matching labels in adata.obs['{broad_label_key}'] "
                "(e.g. ['T_cell', 'B_cell', 'NK'])."
            )
        
        from scrna_pipeline.steps.infercnv import InferCNVStep
        
        steps.append(
            StepSpec(
                name="infercnv",
                step=InferCNVStep(
                    reference_key=broad_label_key,
                    reference_categories=infercnv_reference_categories,
                    layer=infercnv_layer,
                    require_log1p=True,
                ),
                mode="required",
                on_error="raise",
            )
        )



    return steps
