from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from anndata import AnnData

from scrna_pipeline.core.pipeline import Step, StepContext
from scrna_pipeline.tumor_cell_classification import cnv


@dataclass(frozen=True)
class InferCNVStep(Step):
    """
    Run CNV inference using infercnvpy via `cnv.run_infercnv`.

    Assumes / enforces (via `run_infercnv` checks):
      - genome-wide genes (n_vars >= ~1000; not HVG-subsetted)
      - expression is library-size normalized + log1p (and NOT batch-corrected),
        checked heuristically when `require_log1p=True`
      - gene positions exist in `adata.var` (chromosome/start/end) or can be added
      - reference (diploid) cells exist for `reference_key`/`reference_categories`

    Notes
    -----
    - inferCNV must use **non-batch-corrected expression**.
    - Results are stored inside `adata` by infercnvpy.
    """

    name: str = "infercnv"
    reference_key: str = "broad_celltype"
    reference_categories: Sequence[str] = ("T_cell", "B_cell")
    layer: str | None = None
    require_log1p: bool = True

    # escape hatch for future kwargs
    params: Optional[Dict[str, Any]] = None

    def run(self, adata: AnnData, ctx: StepContext) -> AnnData:
        kwargs = dict(self.params or {})

        return cnv.run_infercnv(
            adata,
            reference_key=self.reference_key,
            reference_cat=list(self.reference_categories),
            layer=self.layer,
            require_log1p=self.require_log1p,
            **kwargs,
        )

    def outputs(self) -> Tuple[str, ...]:
        # infercnvpy writes results into adata, but exact keys can vary by version.
        # Set to exact keys once confirmed in your environment.
        return tuple()
