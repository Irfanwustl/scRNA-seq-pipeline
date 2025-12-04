import pandas as pd
import anndata as ad
import scipy.sparse as sp
from pathlib import Path
from typing import Union, Callable, Optional, List


# ============================================================
# Internal helpers (single-responsibility building blocks)
# ============================================================

def _detect_separator(raw_file: str) -> str:
    """
    Decide which separator to use based on file extension.

    - .csv / .csv.gz → ','
    - everything else → '\\t' (tab)

    This keeps backward compatibility for GEO-style RAW .txt/.tsv
    while supporting .csv/.csv.gz for per-sample matrices.
    """
    name = Path(raw_file).name.lower()
    if name.endswith(".csv") or name.endswith(".csv.gz"):
        return ","
    return "\t"


def _read_raw_matrix(raw_file: str) -> pd.DataFrame:
    """
    Read a raw count matrix into a DataFrame with row index.

    Uses index_col=0 and an auto-chosen separator.
    """
    print(f"📌 Loading RAW matrix: {raw_file}")
    sep = _detect_separator(raw_file)
    df = pd.read_csv(raw_file, sep=sep, index_col=0)
    n_rows, n_cols = df.shape
    print(f"   Detected sep={repr(sep)} → Original shape: {n_rows} rows × {n_cols} cols")
    return df


def _infer_genes_are_rows(
    n_rows: int,
    n_cols: int,
    genes_are_rows: Union[str, bool]
) -> bool:
    """
    Decide whether genes are rows or columns.

    Returns
    -------
    True  → genes are rows, cells are columns
    False → cells are rows, genes are columns
    """
    if genes_are_rows == "auto":
        # Heuristic for typical scRNA-seq:
        if (n_rows >= 5000 and n_rows > n_cols * 1.2):
            inferred = True
        elif (n_cols >= 5000 and n_cols > n_rows * 1.2):
            inferred = False
        else:
            raise ValueError(
                "Could not confidently infer orientation (rows vs cols).\n"
                f"Shape=({n_rows}, {n_cols}). Please call with "
                "genes_are_rows=True or False."
            )
    else:
        inferred = bool(genes_are_rows)

    if inferred:
        print("🔍 Using orientation: GENES are rows, CELLS are columns → transposing.")
    else:
        print("🔍 Using orientation: CELLS are rows, GENES are columns → no transpose.")

    return inferred


def _to_cells_by_genes(df: pd.DataFrame, genes_are_rows: bool) -> pd.DataFrame:
    """
    Take a raw matrix and ensure it is in cells × genes orientation.
    """
    if genes_are_rows:
        data = df.T.copy()
    else:
        data = df.copy()

    # Ensure names are strings (AnnData prefers string obs/var names).
    data.index = data.index.astype(str)
    data.columns = data.columns.astype(str)

    print(f"   Final matrix shape (cells × genes): {data.shape}")
    return data


def _build_adata_from_cells_by_genes(data: pd.DataFrame) -> ad.AnnData:
    """
    Build an AnnData from a cells × genes dense DataFrame.

    - .X is CSR sparse
    - .layers['counts'] stores a copy of raw counts
    """
    X = sp.csr_matrix(data.values)
    print("   Converted to sparse CSR matrix.")

    adata = ad.AnnData(X=X)
    adata.obs_names = data.index
    adata.var_names = data.columns

    adata.layers["counts"] = adata.X.copy()

    print(f"✅ AnnData created: {adata.n_obs} cells × {adata.n_vars} genes")
    print("   → .X stores raw counts")
    print("   → .layers['counts'] stores a copy of raw counts")
    print("   → .raw is NOT set (reserve it for normalized matrix later)")

    return adata


def _add_sample_from_obs_names(
    adata: ad.AnnData,
    sample_from_obs_names: Union[None, str, Callable[[str], str]]
) -> None:
    """
    Optionally create adata.obs['sample'] from obs_names.
    """
    if sample_from_obs_names is None:
        return

    print("🧬 Extracting sample information from obs_names...")

    if sample_from_obs_names == "suffix":
        adata.obs["sample"] = adata.obs_names.str.split("-").str[-1]
    elif sample_from_obs_names == "prefix":
        adata.obs["sample"] = adata.obs_names.str.split("-").str[0]
    elif callable(sample_from_obs_names):
        adata.obs["sample"] = adata.obs_names.map(sample_from_obs_names)
    else:
        raise ValueError(
            "sample_from_obs_names must be one of: "
            "None, 'suffix', 'prefix', or a callable(name) -> str."
        )

    print("   'sample' column created in adata.obs. Value counts (top 10):")
    print(adata.obs["sample"].value_counts().head(10))


def _sample_id_from_filename(
    basename: str,
    sample_from_filename: Union[None, str, Callable[[str], str]]
) -> Optional[str]:
    """
    Strategy for deriving a sample ID from a filename.

    - None  → return None
    - 'gsm' → first underscore-separated token (GSM5226574_C51ctr_... → GSM5226574)
    - 'stem'→ basename without extensions (.csv/.csv.gz)
    - callable(basename) → use custom logic
    """
    if sample_from_filename is None:
        return None

    if callable(sample_from_filename):
        return sample_from_filename(basename)

    if sample_from_filename == "gsm":
        return basename.split("_")[0]

    if sample_from_filename == "stem":
        name = basename
        if name.endswith(".gz"):
            name = name[:-3]
        if name.endswith(".csv"):
            name = name[:-4]
        return name

    raise ValueError(
        "sample_from_filename must be one of: "
        "None, 'gsm', 'stem', or a callable(basename) -> str."
    )


# ============================================================
# Public API 1: single RAW matrix → AnnData
# (interface unchanged)
# ============================================================

def raw_matrix_to_h5ad(
    raw_file: str,
    output_h5ad: Optional[str] = None,
    genes_are_rows: Union[str, bool] = "auto",
    sample_from_obs_names: Union[None, str, Callable[[str], str]] = None,
):
    """
    Convert a GEO-style RAW scRNA-seq text matrix into a proper AnnData (.h5ad) file.

    Parameters
    ----------
    raw_file : str
        Path to the RAW matrix file (e.g., 'GSMxxxx_data.raw.matrix.txt.gz').
        File must be a delimited matrix with genes × cells or cells × genes.
        (Tab-delimited is typical for GEO, but .csv/.csv.gz is also supported.)

    output_h5ad : str, optional
        Output filename for the h5ad file.
        If provided, the function saves the resulting AnnData to this path.
        If None (default), no file is written; the AnnData object is just returned.

    genes_are_rows : {'auto', True, False}, default='auto'
        Controls how to interpret the input matrix orientation.

    sample_from_obs_names : {None, 'suffix', 'prefix', callable}, default=None
        Optional helper to derive a "sample" column in `adata.obs` from cell names.

    Returns
    -------
    adata : anndata.AnnData
    """

    df = _read_raw_matrix(raw_file)
    n_rows, n_cols = df.shape

    # Decide orientation
    inferred_genes_are_rows = _infer_genes_are_rows(n_rows, n_cols, genes_are_rows)

    # Ensure cells × genes
    data = _to_cells_by_genes(df, inferred_genes_are_rows)

    # Build AnnData
    adata = _build_adata_from_cells_by_genes(data)

    # Optional: derive 'sample' from obs_names
    _add_sample_from_obs_names(adata, sample_from_obs_names)

    # Save h5ad file only if requested
    if output_h5ad:
        adata.write(output_h5ad)
        print(f"🎉 Saved h5ad file → {output_h5ad}")
    else:
        print("💡 Note: output_h5ad was not provided → skipping file save.")

    return adata


# ============================================================
# Public API 2: folder of per-sample matrices → combined AnnData
# (interface unchanged)
# ============================================================

def folder_raw_matrices_to_h5ad(
    folder: str,
    output_h5ad: Optional[str] = None,
    genes_are_rows: Union[str, bool] = "auto",
    sample_from_filename: Union[None, str, Callable[[str], str]] = "gsm",
):
    """
    Convert a folder of per-sample RAW scRNA-seq matrices (.csv/.csv.gz, .txt/.tsv)
    into one combined AnnData (cells × genes).

    Each file becomes one "sample" in adata.obs["sample"].
    Cells are concatenated and obs_names are made unique.
    """

    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise ValueError(f"{folder!r} is not a directory.")

    # Collect files (you can tweak patterns if needed)
    files: List[Path] = sorted(
        f for f in folder_path.iterdir()
        if f.is_file()
        and (
            f.name.endswith(".csv")
            or f.name.endswith(".csv.gz")
            or f.name.endswith(".txt")
            or f.name.endswith(".txt.gz")
            or f.name.endswith(".tsv")
            or f.name.endswith(".tsv.gz")
        )
    )
    if not files:
        raise ValueError(f"No raw matrix files found under {folder!r}.")

    adatas: List[ad.AnnData] = []
    first_genes = None

    for fp in files:
        print("\n==============================")
        print(f"Processing file: {fp.name}")

        # Reuse the same building blocks as raw_matrix_to_h5ad
        df = _read_raw_matrix(str(fp))
        n_rows, n_cols = df.shape
        inferred_genes_are_rows = _infer_genes_are_rows(n_rows, n_cols, genes_are_rows)
        data = _to_cells_by_genes(df, inferred_genes_are_rows)
        adata = _build_adata_from_cells_by_genes(data)

        # Ensure var_names unique (good hygiene before concat)
        adata.var_names_make_unique()

        # Track gene ordering and consistency
        if first_genes is None:
            first_genes = adata.var_names.copy()
        else:
            if not first_genes.equals(adata.var_names):
                raise ValueError(
                    f"Gene names/order differ between files. "
                    f"Mismatch found in {fp.name}. "
                    "You may need to align genes manually before concatenation."
                )

        # Add 'sample' and 'file' columns
        basename = fp.name
        sample_id = _sample_id_from_filename(basename, sample_from_filename)
        if sample_id is not None:
            adata.obs["sample"] = sample_id
        adata.obs["file"] = basename

        adatas.append(adata)

    print("\n🔗 Concatenating using ad.concat ...")

    # Concatenate all AnnData objects by cells
    adata_all = ad.concat(
        adatas,
        axis=0,
        join="outer",
        label="batch",      # 'batch' column shows which chunk each came from
        keys=None,
        index_unique=None,  # we'll handle uniqueness explicitly
    )

    # Make obs_names unique to avoid collisions
    adata_all.obs_names_make_unique()

    # Ensure raw counts preserved (concat already kept layers, but we set explicitly)
    adata_all.layers["counts"] = adata_all.X.copy()

    print(
        f"✅ Combined AnnData: {adata_all.n_obs} cells × {adata_all.n_vars} genes "
        f"(from {len(adatas)} files)."
    )

    if output_h5ad:
        adata_all.write(output_h5ad)
        print(f"🎉 Saved combined h5ad → {output_h5ad}")
    else:
        print("💡 output_h5ad not provided → returning AnnData without saving.")

    return adata_all
