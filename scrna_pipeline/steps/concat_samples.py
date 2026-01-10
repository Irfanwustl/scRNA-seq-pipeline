"""
Per-sample cleanup and safe concatenation utilities for scRNA-seq AnnData objects.

This module provides a **strict, allowlist-based workflow** for preparing
per-sample scRNA-seq `.h5ad` files for downstream integration or joint analysis.

Core responsibilities
---------------------
1. **Per-sample artifact stripping**
   Remove analysis-specific results that should *never* be reused across samples
   (e.g. PCA, UMAP, neighbors, clustering, HVG annotations), while preserving
   biologically meaningful annotations and expression layers.

2. **Robust sample-aware concatenation**
   Safely concatenate multiple per-sample AnnData objects with:
   - explicit sample identity checks
   - automatic resolution of obs_name collisions
   - strict sanity checks on cell counts and layers

3. **Input flexibility**
   Supports concatenation from:
   - in-memory AnnData objects, and/or
   - a folder of processed `.h5ad` files

Design principles
-----------------
- **Allowlist over denylist**:
  Only explicitly requested fields are kept; everything else is removed.
  This prevents silent reuse of stale or incompatible analysis artifacts.

- **No dependence on upstream processing assumptions**:
  The code does not assume how per-sample preprocessing was performed
  (e.g. whether obs_names were prefixed earlier).

- **Fail fast, fail loud**:
  Any inconsistency in sample identity, cell counts, layer shapes, or naming
  collisions results in an explicit error.

- **Scanpy-agnostic**:
  Operates directly on AnnData objects without depending on Scanpy pipelines.

Typical usage
-------------
Combine a directory of per-sample `.h5ad` files:

>>> combined, reports = combine_samples(
...     folder="per_sample_h5ads/",
...     keep_obs=("broad_celltype",),
... )

Or combine already-loaded AnnData objects:

>>> combined, reports = combine_samples(
...     adatas=[adata1, adata2, adata3]
... )

What this module intentionally does NOT do
------------------------------------------
- No batch correction
- No integration (Harmony, BBKNN, etc.)
- No PCA / UMAP / clustering
- No gene filtering or normalization

Those steps should be performed *after* concatenation.

This module’s sole purpose is to produce a **clean, minimal, and trustworthy**
combined AnnData object suitable for downstream multi-sample analysis.
"""








from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence, Any, Dict, Optional, Tuple

import anndata as ad
from anndata import AnnData
from pathlib import Path




@dataclass(frozen=True)
class StripReport:
    """Summary of what was removed from one AnnData (allowlist-based)."""
    sample: str
    removed_obs_cols: tuple[str, ...]
    removed_var_cols: tuple[str, ...]
    removed_uns: tuple[str, ...]
    removed_layers: tuple[str, ...]
    removed_obsm: tuple[str, ...]
    removed_obsp: tuple[str, ...]


def strip_per_sample_artifacts(
    adata: AnnData,
    *,
    sample: str | None = None,
    # allowlists (keep-only)
    keep_obs: Iterable[str] = ("broad_celltype",),
    keep_var: Iterable[str] = (),  # <- UPDATED: keep no var columns by default
    keep_uns: Iterable[str] = (),  # <- UPDATED: keep nothing in uns by default
    keep_layers: Iterable[str] = ("counts", "log1p"),
    keep_obsm: Iterable[str] = (),  # drop embeddings by default
    keep_obsp: Iterable[str] = (),  # drop graphs by default
    # convenience behaviors
    keep_raw: bool = False,
    keep_broad_scores: bool = True,  # keep broad_*_score if broad_celltype is kept
    keep_id_obs_cols: bool = True,   # keep sample/batch/sample_id/file if present
    # hard-drop knobs
    drop_varm_varp: bool = True,
) -> tuple[AnnData, StripReport]:
    """
    Allowlist cleaner: keep ONLY the specified keys, drop everything else.

    Designed for per-sample objects before concatenation/integration.

    Recommended defaults keep:
      - obs: broad_celltype (+ broad_*_score if keep_broad_scores=True)
             plus id columns if keep_id_obs_cols=True
      - layers: counts, log1p
      

    Drops by default:
      - all embeddings (X_pca, X_umap, etc)
      - all neighbor graphs
      - all uns blobs (pca/neighbors/umap/leiden/scrublet/scrna_pipeline/etc)
      - all var columns (including mt/ribo) unless explicitly allowed
      - varm/varp (optional hard-drop)
      - raw: dropped by default
    """
    a = adata.copy()

    # ---- sample name (no pandas FutureWarning) ----
    if sample is not None:
        sample_name = str(sample)
    else:
        if isinstance(a.uns.get("sample", None), str):
            sample_name = a.uns["sample"]
        elif "sample" in a.obs.columns and a.n_obs > 0:
            sample_name = str(a.obs["sample"].astype(str).iloc[0])
        else:
            sample_name = "unknown"

    keep_obs_set = set(keep_obs)
    keep_var_set = set(keep_var)
    keep_uns_set = set(keep_uns)
    keep_layers_set = set(keep_layers)
    keep_obsm_set = set(keep_obsm)
    keep_obsp_set = set(keep_obsp)

    # Convenience: keep id columns if present
    if keep_id_obs_cols:
        for c in ("sample", "batch", "sample_id", "file"):
            if c in a.obs.columns:
                keep_obs_set.add(c)

    # Convenience: keep broad_*_score if broad_celltype is kept
    if keep_broad_scores and "broad_celltype" in keep_obs_set:
        for c in a.obs.columns:
            if c.startswith("broad_") and c.endswith("_score"):
                keep_obs_set.add(c)

    # ---- OBS ----
    removed_obs_cols = tuple(sorted([c for c in a.obs.columns if c not in keep_obs_set]))
    if removed_obs_cols:
        a.obs = a.obs.drop(columns=list(removed_obs_cols))

    # ---- VAR ----
    removed_var_cols = tuple(sorted([c for c in a.var.columns if c not in keep_var_set]))
    if removed_var_cols:
        a.var = a.var.drop(columns=list(removed_var_cols))

    # ---- UNS ----
    removed_uns = tuple(sorted([k for k in list(a.uns.keys()) if k not in keep_uns_set]))
    for k in removed_uns:
        del a.uns[k]

    # ---- LAYERS ----
    removed_layers = tuple(sorted([k for k in list(a.layers.keys()) if k not in keep_layers_set]))
    for k in removed_layers:
        del a.layers[k]

    # ---- OBS M / OBS P ----
    removed_obsm = tuple(sorted([k for k in list(a.obsm.keys()) if k not in keep_obsm_set]))
    for k in removed_obsm:
        del a.obsm[k]

    removed_obsp = tuple(sorted([k for k in list(a.obsp.keys()) if k not in keep_obsp_set]))
    for k in removed_obsp:
        del a.obsp[k]

    # ---- VARM / VARP ----
    if drop_varm_varp:
        for k in list(a.varm.keys()):
            del a.varm[k]
        for k in list(a.varp.keys()):
            del a.varp[k]

    # ---- RAW ----
    if not keep_raw:
        a.raw = None

    report = StripReport(
        sample=sample_name,
        removed_obs_cols=removed_obs_cols,
        removed_var_cols=removed_var_cols,
        removed_uns=removed_uns,
        removed_layers=removed_layers,
        removed_obsm=removed_obsm,
        removed_obsp=removed_obsp,
    )
    return a, report



# -----------------------------
# Small helpers (single-purpose)
# -----------------------------

def _load_h5ads_from_folder(
    folder: str | Path,
    *,
    pattern: str = "*.h5ad",
    sort: bool = True,
) -> list[AnnData]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")

    paths = list(folder.glob(pattern))
    if not paths:
        raise ValueError(f"No files matching {pattern!r} found in {folder}")

    if sort:
        paths = sorted(paths)

    return [ad.read_h5ad(p) for p in paths]

def get_sample_id(a: AnnData, sample_key: str) -> str:
    """Return the sample id (must exist and be constant within the AnnData)."""
    if sample_key not in a.obs.columns:
        raise ValueError(f"Missing required obs column {sample_key!r}.")

    s = a.obs[sample_key].astype(str)
    sid = str(s.iloc[0])
    if (s != sid).any():
        raise ValueError(
            f"obs[{sample_key!r}] must be constant within a sample, but found multiple values."
        )
    return sid


def has_cross_sample_obs_collisions(adatas: Sequence[AnnData]) -> bool:
    """True if any obs_names are shared across different AnnData objects."""
    seen: set[str] = set()
    for a in adatas:
        for n in a.obs_names:
            if n in seen:
                return True
            seen.add(n)
    return False


def prefix_obs_names(a: AnnData, prefix: str, delim: str = ":") -> AnnData:
    """Return a copy with prefixed obs_names."""
    b = a.copy()
    b.obs_names = [f"{prefix}{delim}{cid}" for cid in b.obs_names]
    return b


def concat_with_sanity(
    adatas: Sequence[AnnData],
    *,
    join: Literal["inner", "outer"] = "outer",
    required_layers: Iterable[str] = ("counts", "log1p"),
    sample_key: str = "sample_id",
    expected_nobs_by_sample: dict[str, int],
) -> AnnData:
    """Concat and run sanity checks right before returning."""
    combined = ad.concat(
        list(adatas),
        axis=0,
        join=join,
        merge="same",
        uns_merge="unique",
        index_unique=None,
    )

    # 1) total cells
    expected_total = int(sum(expected_nobs_by_sample.values()))
    if int(combined.n_obs) != expected_total:
        raise RuntimeError(
            f"Concat cell-count mismatch: got {combined.n_obs}, expected {expected_total}."
        )

    # 2) unique obs_names
    if not combined.obs_names.is_unique:
        raise RuntimeError("Concat produced non-unique obs_names.")

    # 3) sample_key present + per-sample counts match
    if sample_key not in combined.obs.columns:
        raise RuntimeError(f"Missing {sample_key!r} in combined.obs after concat.")

    got_counts = combined.obs[sample_key].astype(str).value_counts().to_dict()
    for sid, expected_n in expected_nobs_by_sample.items():
        got_n = int(got_counts.get(sid, 0))
        if got_n != int(expected_n):
            raise RuntimeError(
                f"Per-sample count mismatch for {sid!r}: got {got_n}, expected {expected_n}."
            )

    # 4) required layers exist and align to combined shape
    for lk in required_layers:
        if lk not in combined.layers:
            raise RuntimeError(f"Missing expected layer {lk!r} after concat.")
        if combined.layers[lk].shape != combined.shape:
            raise RuntimeError(
                f"Layer {lk!r} shape {combined.layers[lk].shape} != combined shape {combined.shape}."
            )

    return combined


# -----------------------------
# Minimal configuration object
# -----------------------------

@dataclass(frozen=True)
class CombineConfig:
    sample_key: str = "sample_id"  # your preferred default
    join: Literal["inner", "outer"] = "outer"
    ensure_unique_obs_names: Literal["auto", "always", "never"] = "auto"
    unique_delim: str = ":"
    required_layers: tuple[str, ...] = ("counts", "log1p")


# -----------------------------
# Main entry point 
# -----------------------------


def combine_samples(
    adatas: Sequence[AnnData] | None = None,
    *,
    folder: str | Path | None = None,
    pattern: str = "*.h5ad",
    cfg: CombineConfig = CombineConfig(),
    # pass-through to your existing strip function (keep-only)
    keep_obs: Iterable[str] = ("broad_celltype",),
    keep_var: Iterable[str] = (),
    keep_uns: Iterable[str] = (),
    keep_layers: Iterable[str] = ("counts", "log1p"),
    keep_broad_scores: bool = True,
    keep_id_obs_cols: bool = True,
) -> tuple[AnnData, list]:
    """
    Clean each sample (via strip_per_sample_artifacts), ensure unique obs_names if needed,
    then concatenate with sanity checks.

    Input can be either:
      - adatas: pre-loaded AnnData objects
      - folder: a directory of processed .h5ad files (pattern-matched)

    Returns (combined, reports).
    """
    # -----------------------------
    # 0) Resolve inputs
    # -----------------------------
    if adatas is None:
        adatas = []

    adatas_list: list[AnnData] = list(adatas)

    if folder is not None:
        adatas_list.extend(_load_h5ads_from_folder(folder, pattern=pattern, sort=True))

    if len(adatas_list) == 0:
        raise ValueError("No AnnData objects provided (adatas empty and/or folder had no matching files).")

    cleaned: list[AnnData] = []
    reports: list = []
    expected_nobs_by_sample: dict[str, int] = {}

    # -----------------------------
    # 1) Clean per sample
    # -----------------------------
    for a in adatas_list:
        sid = get_sample_id(a, cfg.sample_key)

        a2, rep = strip_per_sample_artifacts(
            a,
            sample=sid,
            keep_obs=keep_obs,
            keep_var=keep_var,
            keep_uns=keep_uns,
            keep_layers=keep_layers,
            keep_obsm=(),
            keep_obsp=(),
            keep_raw=False,  # you said you dropped .raw
            keep_broad_scores=keep_broad_scores,
            keep_id_obs_cols=keep_id_obs_cols,
        )

        # guarantee sample_id present post-strip
        a2.obs[cfg.sample_key] = sid

        if not a2.obs_names.is_unique:
            raise ValueError(f"Within-sample obs_names are not unique for sample {sid!r}.")

        if sid in expected_nobs_by_sample:
            raise ValueError(
                f"Duplicate sample id detected: {sid!r}. "
                f"Each input h5ad must have a unique obs[{cfg.sample_key!r}] value."
            )

        expected_nobs_by_sample[sid] = int(a2.n_obs)
        cleaned.append(a2)
        reports.append(rep)

    # -----------------------------
    # 2) Ensure global uniqueness (robust)
    # -----------------------------
    collisions = has_cross_sample_obs_collisions(cleaned)

    if cfg.ensure_unique_obs_names == "never" and collisions:
        raise ValueError(
            "obs_names collide across samples and ensure_unique_obs_names='never'. "
            "Set it to 'auto' or 'always'."
        )

    do_prefix = (cfg.ensure_unique_obs_names == "always") or (
        cfg.ensure_unique_obs_names == "auto" and collisions
    )

    if do_prefix:
        cleaned = [
            prefix_obs_names(a, get_sample_id(a, cfg.sample_key), cfg.unique_delim)
            for a in cleaned
        ]

    # -----------------------------
    # 3) Concat + sanity
    # -----------------------------
    combined = concat_with_sanity(
        cleaned,
        join=cfg.join,
        required_layers=cfg.required_layers,
        sample_key=cfg.sample_key,
        expected_nobs_by_sample=expected_nobs_by_sample,
    )

    return combined, reports






@dataclass(frozen=True)
class CombineSamplesStep:
    """
    Step wrapper for `combine_samples`.

    Purpose
    -------
    Cleans each per-sample AnnData using an allowlist strategy, ensures globally
    unique `obs_names` if needed, and concatenates with strict sanity checks.

    Input modes
    ----------
    - In-memory: pass `adatas=[...]` via params (rare in pipeline steps)
    - File-based: pass `folder=...` (recommended for batch runners)

    Output
    ------
    Returns a *single* combined AnnData (cells concatenated).
    Stores strip/concat reports in:
      `adata.uns["scrna_pipeline"]["combine_samples"]["reports"]`

    Notes
    -----
    - This step is typically used at the "integration boundary":
        per-sample preprocessing -> save -> (this step) combine -> batch correction -> neighbors/UMAP
    - The "reports" may be large; if that becomes an issue, you can store a
      summary instead (counts of removed keys, etc.).
    """

    name: str = "combine_samples"

    # Recommended usage: read from folder of .h5ad files
    folder: str | None = None
    pattern: str = "*.h5ad"

    # Combine behavior
    cfg: Any = None  # if None, CombineConfig() will be used inside run()

    # Allowlist passthroughs to strip_per_sample_artifacts via combine_samples
    keep_obs: Tuple[str, ...] = ("broad_celltype",)
    keep_var: Tuple[str, ...] = ()
    keep_uns: Tuple[str, ...] = ()
    keep_layers: Tuple[str, ...] = ("counts", "log1p")
    keep_broad_scores: bool = True
    keep_id_obs_cols: bool = True

    # Extra kwargs forwarded to combine_samples (rare)
    params: Optional[Dict[str, Any]] = None

    def run(self, adata: AnnData, ctx) -> AnnData:
        """
        This step ignores the incoming `adata` and produces a new combined AnnData.
        The incoming object is treated as a placeholder to match the Step interface.
        """
        kwargs = dict(self.params or {})

        cfg = self.cfg if self.cfg is not None else CombineConfig()

        if self.folder is None and "adatas" not in kwargs:
            raise ValueError(
                "CombineSamplesStep requires either `folder=...` or `params={'adatas': [...]}`."
            )

        combined, reports = combine_samples(
            # by default we want folder-driven usage; but allow adatas override via params
            adatas=kwargs.pop("adatas", None),
            folder=self.folder,
            pattern=self.pattern,
            cfg=cfg,
            keep_obs=self.keep_obs,
            keep_var=self.keep_var,
            keep_uns=self.keep_uns,
            keep_layers=self.keep_layers,
            keep_broad_scores=self.keep_broad_scores,
            keep_id_obs_cols=self.keep_id_obs_cols,
            **kwargs,
        )

        # Store provenance
        combined.uns.setdefault("scrna_pipeline", {})
        combined.uns["scrna_pipeline"].setdefault("combine_samples", {})
        combined.uns["scrna_pipeline"]["combine_samples"].update(
            {
                "folder": self.folder,
                "pattern": self.pattern,
                "cfg": {
                    "sample_key": cfg.sample_key,
                    "join": cfg.join,
                    "ensure_unique_obs_names": cfg.ensure_unique_obs_names,
                    "unique_delim": cfg.unique_delim,
                    "required_layers": tuple(cfg.required_layers),
                },
                "keep_obs": tuple(self.keep_obs),
                "keep_var": tuple(self.keep_var),
                "keep_uns": tuple(self.keep_uns),
                "keep_layers": tuple(self.keep_layers),
                "keep_broad_scores": bool(self.keep_broad_scores),
                "keep_id_obs_cols": bool(self.keep_id_obs_cols),
                "reports": reports,  # list[StripReport]
            }
        )

        return combined

    def outputs(self) -> Tuple[str, ...]:
        return ("combined_adata", "uns[scrna_pipeline/combine_samples/reports]")

