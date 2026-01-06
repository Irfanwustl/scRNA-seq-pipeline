from __future__ import annotations
from pathlib import Path
import urllib.request
import gzip, shutil
import pandas as pd
import infercnvpy as infercnv
import numpy as np
from scipy.sparse import issparse
from typing import Any, Dict, Sequence
from anndata import AnnData






def _download_gencode_gtf(
    *,
    release: int = 44,
    cache_dir: str | Path = "~/.cache/genome/gtf",
    verbose: bool = True,
) -> Path:
    """
    Download and cache GENCODE human GTF (hg38).
    Returns path to uncompressed .gtf file.
    """
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    gz_name = f"gencode.v{release}.annotation.gtf.gz"
    gtf_name = f"gencode.v{release}.annotation.gtf"

    gz_path = cache_dir / gz_name
    gtf_path = cache_dir / gtf_name

    url = (
        f"https://ftp.ebi.ac.uk/pub/databases/gencode/"
        f"Gencode_human/release_{release}/{gz_name}"
    )

    if gtf_path.exists():
        if verbose:
            print(f"[GTF] Using cached GTF: {gtf_path}")
        return gtf_path

    if verbose:
        print(f"[GTF] Downloading GENCODE v{release} from:\n{url}")

    urllib.request.urlretrieve(url, gz_path)

    if verbose:
        print(f"[GTF] Extracting {gz_path.name}")

    with gzip.open(gz_path, "rb") as f_in, open(gtf_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    gz_path.unlink(missing_ok=True)

    if verbose:
        print(f"[GTF] Ready: {gtf_path}")

    return gtf_path


def add_gene_positions_from_gtf(
    adata,
    gtf_path: str | Path | None = None,
    *,
    gencode_release: int = 44,
    gene_symbol_key: str | None = None,
    make_var_unique: bool = True,
    cache_dir: str | Path = "~/.cache/genome/gtf",
    verbose: bool = True,
):
    """
    Add `chromosome`, `start`, `end` to adata.var.
    Automatically downloads GENCODE GTF if gtf_path is None.

    Parameters
    ----------
    adata : AnnData
    gtf_path : str or Path or None
        Path to GTF. If None, downloads GENCODE human GTF (hg38).
    gencode_release : int
        GENCODE release to download if gtf_path is None.
    gene_symbol_key : str, optional
        Column in adata.var containing gene symbols.
        If None, adata.var_names are used.
    cache_dir : str or Path
        Where to cache downloaded GTF.
    """
    if make_var_unique:
        adata.var_names_make_unique()

    # ---- get GTF path (download if needed) ----
    if gtf_path is None:
        gtf_path = _download_gencode_gtf(
            release=gencode_release,
            cache_dir=cache_dir,
            verbose=verbose,
        )
    else:
        gtf_path = Path(gtf_path).expanduser().resolve()

    if verbose:
        print(f"[add_gene_positions] Reading GTF: {gtf_path}")

    # ---- read minimal GTF ----
    cols = [
        "chromosome", "source", "feature",
        "start", "end", "score", "strand", "frame", "attribute"
    ]
    gtf = pd.read_csv(
        gtf_path,
        sep="\t",
        comment="#",
        header=None,
        names=cols,
        dtype={"chromosome": str},
    )

    gtf = gtf[gtf["feature"] == "gene"].copy()

    def _get_attr(attr: str, key: str):
        try:
            return attr.split(f'{key} "')[1].split('"')[0]
        except IndexError:
            return None

    gtf["gene_name"] = gtf["attribute"].apply(lambda x: _get_attr(x, "gene_name"))
    gtf["gene_id"] = gtf["attribute"].apply(lambda x: _get_attr(x, "gene_id"))

    gtf = gtf.dropna(subset=["gene_name"])
    gtf = gtf.sort_values(["gene_name", "start"]).drop_duplicates("gene_name")

    gene_pos = gtf.set_index("gene_name")[["chromosome", "start", "end"]]

    # ---- map to adata.var ----
    if gene_symbol_key is None:
        genes = pd.Index(adata.var_names.astype(str))
    else:
        if gene_symbol_key not in adata.var.columns:
            raise KeyError(f"{gene_symbol_key} not found in adata.var.columns")
        genes = pd.Index(adata.var[gene_symbol_key].astype(str))

    mapped = gene_pos.reindex(genes)

    adata.var["chromosome"] = mapped["chromosome"].values
    adata.var["start"] = mapped["start"].values
    adata.var["end"] = mapped["end"].values

    if verbose:
        missing = adata.var["chromosome"].isna().sum()
        print(
            f"[add_gene_positions] Positions added. "
            f"Missing genes: {missing}/{adata.n_vars}"
        )

    return adata





def run_infercnv(
    adata,
    *,
    reference_key: str | None = None,
    reference_cat: str | list[str] | None = None,
    layer: str | None = None,
    require_log1p: bool = True,
):
    """
    Run CNV inference using infercnvpy on an AnnData object.

    ============================================================
    DATA REQUIREMENTS (CRITICAL – READ BEFORE USING)
    ============================================================
    infercnvpy assumes the following state of the AnnData object:

    1) Expression values
       - Must be library-size normalized
       - Must be log1p-transformed
       - Must NOT be batch-corrected
       - Must contain ALL genes (not HVG-subsetted)

       Typical preparation:
           sc.pp.normalize_total(adata, target_sum=1e4)
           sc.pp.log1p(adata)

    2) Gene annotations
       adata.var MUST contain:
           - chromosome
           - start
           - end

    3) Reference (diploid) cells (strongly recommended)
       - Provided via `reference_key` and `reference_cat`
       - Usually immune or stromal populations

    4) Layer usage
       - By default, inferCNV uses `adata.X`
       - If your log1p data is stored elsewhere (e.g. `layers["log1p"]`),
         pass `layer="log1p"`

    ⚠️ Batch-corrected embeddings or expression matrices (Harmony,
       Scanorama, ComBat, etc.) MUST NOT be used.

    ============================================================

    Parameters
    ----------
    adata : AnnData
        Annotated single-cell object. Modified IN PLACE.

    reference_key : str, optional
        Column in `adata.obs` used to identify reference cells.

    reference_cat : str or list[str], optional
        Category or categories in `reference_key` defining diploid cells.

    layer : str, optional
        Expression layer to use instead of `adata.X`.
        Example: layer="log1p"

    require_log1p : bool, default True
        If True, performs a heuristic check that the data appears log1p-like
        and raises an error if not.

    Returns
    -------
    None
        inferCNV results are stored inside `adata`.
    """

    # ------------------------------------------------------------
    # Step 0: Basic structural checks
    # ------------------------------------------------------------
    if adata.n_vars < 1000:
        raise ValueError(
            f"inferCNV requires genome-wide genes. "
            f"Found only {adata.n_vars} genes. "
            f"Did you subset to HVGs?"
        )

    if layer is not None and layer not in adata.layers:
        raise KeyError(f"Requested layer '{layer}' not found in adata.layers")

    X = adata.layers[layer] if layer is not None else adata.X



    # ------------------------------------------------------------
    # Step 1: Heuristic check for log1p-like data (sparse-safe)
    # ------------------------------------------------------------
    if require_log1p:

        def _sparse_min_max(M):
            """
            Return (min, max) for dense or sparse matrices.

            Notes for sparse matrices:
            - implicit zeros are NOT stored in M.data
            - so the true minimum is min(0, min(data)) unless matrix is all-zero
            - the true maximum is max(0, max(data)) unless matrix is all-zero
            """
            if issparse(M):
                data = M.data  # stored non-zeros only
                if data.size == 0:
                    return 0.0, 0.0
                if np.isnan(data).any():
                    raise ValueError("Expression matrix contains NaNs (in sparse .data).")
                mn = float(min(0.0, data.min()))
                mx = float(max(0.0, data.max()))
                return mn, mx
            else:
                arr = np.asarray(M)
                if np.isnan(arr).any():
                    raise ValueError("Expression matrix contains NaNs.")
                return float(arr.min()), float(arr.max())

        min_val, max_val = _sparse_min_max(X)

        # Very loose sanity bounds for log1p-normalized scRNA data:
        # - values should be >= 0 (allow tiny negative numerical noise)
        # - values should not be extremely large (raw counts can be 100s+)
        if max_val > 50 or min_val < -1e-6:
            raise ValueError(
                "Expression values do not appear to be log1p-normalized.\n"
                f"Observed range: min={min_val:.2f}, max={max_val:.2f}\n"
                "Please ensure you have run normalize_total + log1p, "
                "and that batch correction has NOT been applied.\n"
                "If your log1p data is stored in a layer, pass `layer='log1p'`."
            )


    # ------------------------------------------------------------
    # Step 2: Ensure gene genomic coordinates exist
    # ------------------------------------------------------------
    required_cols = {"chromosome", "start", "end"}
    if not required_cols.issubset(adata.var.columns):
        adata = add_gene_positions_from_gtf(adata)

    # ------------------------------------------------------------
    # Step 3: Validate reference cells (if provided)
    # ------------------------------------------------------------
    if reference_key is not None:
        if reference_key not in adata.obs:
            raise KeyError(f"{reference_key} not found in adata.obs")

        if reference_cat is None:
            raise ValueError(
                "reference_key was provided but reference_cat is None.\n"
                "Please specify which categories represent diploid cells."
            )

        ref_cats = (
            [reference_cat]
            if isinstance(reference_cat, str)
            else list(reference_cat)
        )

        missing = set(ref_cats) - set(adata.obs[reference_key].unique())
        if missing:
            raise ValueError(
                f"Reference categories not found in {reference_key}: {missing}"
            )

    # ------------------------------------------------------------
    # Step 4: Run inferCNV
    # ------------------------------------------------------------
    infercnv.tl.infercnv(
        adata,
        reference_key=reference_key,
        reference_cat=reference_cat,
        layer=layer,
    )









def run_infercnv_per_batch(
    adata: AnnData,
    *,
    batch_key: str = "sample",
    reference_key: str | None = None,
    reference_cat: str | list[str] | None = None,
    layer: str | None = "log1p",
    require_log1p: bool = True,
    infercnv_kwargs: Dict[str, Any] | None = None,
    # naming / storage
    key_sep: str = "__",
    store_uns_key: str = "infercnv_per_batch",
    store_obsm_in_uns: bool = True,
    store_uns_in_uns: bool = True,
    verbose: bool = True,
) -> AnnData:
    """
    Run inferCNV separately for each batch and write results back into `adata`
    using batch-prefixed keys.

    Storage strategy (memory-efficient)
    -----------------------------------
    - NEW per-cell (1D) outputs produced by infercnvpy in `sub.obs` are written into
      `adata.obs` with batch-prefixed column names:
          adata.obs[f"{batch}__{obs_key}"]

    - NEW multi-dimensional outputs produced by infercnvpy in `sub.obsm` are stored
      in `adata.uns[store_uns_key][batch]["obsm"][obsm_key]` (default).
      This avoids allocating huge (adata.n_obs x k) matrices.

    - NEW infercnvpy outputs in `sub.uns` are stored in
      `adata.uns[store_uns_key][batch]["uns"][uns_key]` (default).

    Reference handling
    ------------------
    If reference_key/reference_cat are provided:
    - use only reference categories present in each batch
    - skip the batch if none are present

    Notes
    -----
    - This wrapper assumes you already prepared non-batch-corrected log1p data
      (e.g., in layer="log1p") and gene positions exist or can be added by run_infercnv().
    - For plotting CNV embeddings later, slice the batch and attach the stored
      obsm to the slice (see helper in docstring below).

    Helper for plotting later
    -------------------------
    >>> b = "S1"
    >>> sub = adata[adata.obs[batch_key].astype(str) == b].copy()
    >>> sub.obsm["X_cnv"] = adata.uns[store_uns_key][b]["obsm"]["X_cnv"]
    >>> sc.pl.embedding(sub, basis="X_cnv", color=[reference_key])
    """
    print("##### irf #########")
    if batch_key not in adata.obs:
        raise KeyError(f"batch_key={batch_key!r} not found in adata.obs")

    infercnv_kwargs = {} if infercnv_kwargs is None else infercnv_kwargs
    batches = adata.obs[batch_key].astype(str).unique().tolist()

    # Ensure container exists
    if store_obsm_in_uns or store_uns_in_uns:
        adata.uns.setdefault(store_uns_key, {})

    for b in batches:
        idx = (adata.obs[batch_key].astype(str) == b).to_numpy()
        if idx.sum() == 0:
            continue

        sub = adata[idx].copy()
        bkey = str(b)

        # ------------------------------------------------------------
        # Reference handling: use only reference categories present in THIS batch
        # ------------------------------------------------------------
        ref_cats_present = None
        if reference_key is not None and reference_cat is not None:
            if reference_key not in sub.obs:
                raise KeyError(f"reference_key={reference_key!r} not found in adata.obs")

            ref_cats = [reference_cat] if isinstance(reference_cat, str) else list(reference_cat)
            present = set(sub.obs[reference_key].astype(str).unique())
            ref_cats_present = [c for c in ref_cats if str(c) in present]

            if len(ref_cats_present) == 0:
                if verbose:
                    print(f"[inferCNV] batch={b} SKIP (no reference cells present)")
                continue

            if verbose:
                if len(ref_cats_present) != len(ref_cats):
                    missing = sorted(set(map(str, ref_cats)) - set(map(str, ref_cats_present)))
                    print(f"[inferCNV] batch={b} refs missing (ok): {missing}")
                print(f"[inferCNV] batch={b} refs_used={list(map(str, ref_cats_present))}")

        if verbose:
            print(f"[inferCNV] batch={b} running (n_cells={sub.n_obs}, n_genes={sub.n_vars})")

        # Snapshot keys BEFORE (to detect what infercnvpy adds)
        obs_before = set(sub.obs.columns)
        obsm_before = set(sub.obsm.keys())
        uns_before = set(sub.uns.keys())

        # ------------------------------------------------------------
        # Run your existing single-AnnData inferCNV (unchanged)
        # ------------------------------------------------------------
        run_infercnv(
            sub,
            reference_key=reference_key,
            reference_cat=ref_cats_present if ref_cats_present is not None else reference_cat,
            layer=layer,
            require_log1p=require_log1p,
            **infercnv_kwargs,
        )

        # Detect keys added by inferCNV
        new_obs = [k for k in sub.obs.columns if k not in obs_before]
        new_obsm = [k for k in sub.obsm.keys() if k not in obsm_before]
        new_uns = [k for k in sub.uns.keys() if k not in uns_before]

        # ------------------------------------------------------------
        # Store NEW obs columns in the full adata (batch-prefixed)
        # ------------------------------------------------------------
        for k in new_obs:
            out_k = f"{bkey}{key_sep}{k}"
            adata.obs.loc[adata.obs.index[idx], out_k] = sub.obs[k].to_numpy()

        # ------------------------------------------------------------
        # Store NEW obsm/uns under adata.uns[store_uns_key][batch] (memory efficient)
        # ------------------------------------------------------------
        if store_obsm_in_uns or store_uns_in_uns:
            bucket = adata.uns[store_uns_key].setdefault(bkey, {})
        else:
            bucket = None

        stored_obsm_keys: list[str] = []
        stored_uns_keys: list[str] = []

        if store_obsm_in_uns and new_obsm:
            obsm_dict = bucket.setdefault("obsm", {})
            for k in new_obsm:
                X = np.asarray(sub.obsm[k])
                obsm_dict[k] = X
                stored_obsm_keys.append(k)

        if store_uns_in_uns and new_uns:
            uns_dict = bucket.setdefault("uns", {})
            for k in new_uns:
                uns_dict[k] = sub.uns[k]
                stored_uns_keys.append(k)

        if verbose:
            obs_out = [f"{bkey}{key_sep}{k}" for k in new_obs]
            print(f"[inferCNV] batch={b} done.")
            print(f"  • obs added:  {obs_out}")
            if store_obsm_in_uns:
                print(f"  • obsm stored in uns[{store_uns_key!r}][{bkey!r}]['obsm']: {stored_obsm_keys}")
            if store_uns_in_uns:
                print(f"  • uns stored in uns[{store_uns_key!r}][{bkey!r}]['uns']:  {stored_uns_keys}")

    return adata
