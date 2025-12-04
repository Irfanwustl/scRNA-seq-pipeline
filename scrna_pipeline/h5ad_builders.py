import pandas as pd
import anndata as ad
import scipy.sparse as sp
from typing import Union, Callable, Optional


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
        File must be a tab-delimited matrix with genes × cells or cells × genes.

    output_h5ad : str, optional
        Output filename for the h5ad file.
        If provided, the function saves the resulting AnnData to this path.
        If None (default), no file is written; the AnnData object is just returned.

    genes_are_rows : {'auto', True, False}, default='auto'
        Controls how to interpret the input matrix orientation.

        - 'auto'
            Use a simple heuristic based on matrix shape:
                * If n_rows >= 5000 and n_rows > 1.2 * n_cols → assume rows are genes.
                * If n_cols >= 5000 and n_cols > 1.2 * n_rows → assume columns are genes.
            If the shape is ambiguous, a ValueError is raised and you must set
            genes_are_rows=True/False explicitly.
        - True
            Force interpretation that rows are genes and columns are cells.
            (The matrix will be transposed to become cells × genes.)
        - False
            Force interpretation that rows are cells and columns are genes.
            (No transpose is applied.)

        In all cases, the final AnnData will have shape (n_cells × n_genes).

    sample_from_obs_names : {None, 'suffix', 'prefix', callable}, default=None
        Optional helper to derive a "sample" column in `adata.obs` from cell names.

        Many datasets encode sample/batch information inside cell barcodes, e.g.:
            - 'AAACCTGAGAGATGAG-1'      (10x lane ID as suffix)
            - 'AAACCTGAGAGATGAG_S1'    (sample ID as suffix after underscore)
            - 'S1-AAACCTGAGAGATGAG'    (sample ID as prefix)

        This argument controls whether/how we parse that information:

        - None
            Do nothing. No "sample" column is added.
        - 'suffix'
            Split each obs_name by '-' and take the last field.
            Use this when the *last* token after '-' corresponds to sample/lane,
            e.g. 'AAACCTGAGAGATGAG-1', 'AAACCTGAGAGATGAG-2'.
        - 'prefix'
            Split each obs_name by '-' and take the first field.
            Use this when the *first* token encodes the sample,
            e.g. 'SampleA-AAACCTGAGAGATGAG', 'SampleB-AAACCTGAGAGATGAG'.
        - callable
            A user-defined function `f(name: str) -> str` applied to each obs_name.
            This is the most flexible option and lets you implement any custom
            parsing logic (e.g. splitting on '_', taking middle fields, etc.).

        The parsed values are stored in `adata.obs["sample"]`.

    Returns
    -------
    adata : anndata.AnnData
        AnnData object with:
        - raw UMI counts in .X  (cells × genes, stored as CSR sparse matrix)
        - a copy of raw counts in .layers["counts"]
        Note: .raw is intentionally NOT set here. The recommended pattern is to
        set `adata.raw = adata` later, after normalization/log-transform and
        before subsetting to HVGs.

    """

    print("📌 Loading RAW matrix:", raw_file)
    df = pd.read_csv(raw_file, sep="\t", index_col=0)
    n_rows, n_cols = df.shape
    print(f"   Original shape: {n_rows} rows × {n_cols} cols")

    # ------------------------------------------------------------
    # Decide orientation of the input matrix
    # ------------------------------------------------------------
    if genes_are_rows == "auto":
        # Heuristic for typical scRNA-seq:
        #   - usually there are many more genes than cells *or* vice versa.
        if (n_rows >= 5000 and n_rows > n_cols * 1.2):
            inferred_genes_are_rows = True
        elif (n_cols >= 5000 and n_cols > n_rows * 1.2):
            inferred_genes_are_rows = False
        else:
            # If dimensions are similar, we refuse to guess.
            raise ValueError(
                "Could not confidently infer orientation (rows vs cols).\n"
                f"Shape={df.shape}. Please call raw_matrix_to_h5ad(..., "
                "genes_are_rows=True or False)."
            )
    else:
        inferred_genes_are_rows = bool(genes_are_rows)

    # ------------------------------------------------------------
    # Apply orientation so that we always end up with cells × genes
    # ------------------------------------------------------------
    if inferred_genes_are_rows:
        print("🔍 Using orientation: GENES are rows, CELLS are columns → transposing.")
        data = df.T.copy()   # cells × genes
    else:
        print("🔍 Using orientation: CELLS are rows, GENES are columns → no transpose.")
        data = df.copy()     # cells × genes

    # Ensure names are strings (AnnData prefers string obs/var names).
    cell_names = data.index.astype(str)
    gene_names = data.columns.astype(str)

    print(f"   Final matrix shape (cells × genes): {data.shape}")

    # ------------------------------------------------------------
    # Convert to sparse matrix for memory efficiency
    # ------------------------------------------------------------
    X = sp.csr_matrix(data.values)
    print("   Converted to sparse CSR matrix.")

    # ------------------------------------------------------------
    # Build AnnData object
    # ------------------------------------------------------------
    adata = ad.AnnData(X=X)
    adata.obs_names = cell_names
    adata.var_names = gene_names

    # CONTRACT: .X and layers["counts"] both contain raw counts
    adata.layers["counts"] = adata.X.copy()

    print(f"✅ AnnData created: {adata.n_obs} cells × {adata.n_vars} genes")
    print("   → .X stores raw counts")
    print("   → .layers['counts'] stores a copy of raw counts")
    print("   → .raw is NOT set (reserve it for normalized matrix later)")

    # ------------------------------------------------------------
    # Optional: derive a 'sample' column from obs_names
    # ------------------------------------------------------------
    if sample_from_obs_names is not None:
        print("🧬 Extracting sample information from obs_names...")

        if sample_from_obs_names == "suffix":
            # Typical 10x style: BARCODE-SAMPLE or BARCODE-LANE
            # e.g. 'AAACCTGAGAGATGAG-1' → sample='1'
            adata.obs["sample"] = adata.obs_names.str.split("-").str[-1]

        elif sample_from_obs_names == "prefix":
            # Prefix style: SAMPLE-BARCODE
            # e.g. 'Tumor1-AAACCTGAGAGATGAG' → sample='Tumor1'
            adata.obs["sample"] = adata.obs_names.str.split("-").str[0]

        elif callable(sample_from_obs_names):
            # Fully customizable parsing:
            #   sample_from_obs_names(name: str) -> str
            adata.obs["sample"] = adata.obs_names.map(sample_from_obs_names)

        else:
            raise ValueError(
                "sample_from_obs_names must be one of: "
                "None, 'suffix', 'prefix', or a callable(name) -> str."
            )

        print("   'sample' column created in adata.obs. Value counts (top 10):")
        print(adata.obs["sample"].value_counts().head(10))

    # ------------------------------------------------------------
    # Save h5ad file only if requested
    # ------------------------------------------------------------
    if output_h5ad:
        adata.write(output_h5ad)
        print(f"🎉 Saved h5ad file → {output_h5ad}")
    else:
        print("💡 Note: output_h5ad was not provided → skipping file save.")

    return adata
