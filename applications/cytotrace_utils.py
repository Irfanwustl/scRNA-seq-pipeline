import numpy as np
import scipy.sparse as sp


def export_cytotrace_format(
    X_cells_by_genes,
    gene_names,
    cell_ids,
    out_path,
    chunk_genes=256,
):
    """
    Export a sparse expression matrix to the tab-delimited format expected by CytoTRACE2.

    Input:
        X_cells_by_genes : scipy sparse matrix of shape (n_cells, n_genes)
        gene_names       : list/array of gene names (length = n_genes)
        cell_ids         : list/array of cell IDs (length = n_cells)
        out_path         : output text file
        chunk_genes      : number of genes to process at a time

    Output format:
            Cell1   Cell2   Cell3   ...
    Gene1     x       x       x
    Gene2     x       x       x
    ...
    """

    # Ensure a sparse matrix is provided to avoid excessive memory usage.
    assert sp.issparse(X_cells_by_genes), "Input must be a scipy sparse matrix."

    n_cells, n_genes = X_cells_by_genes.shape

    # Validate metadata.
    assert len(gene_names) == n_genes
    assert len(cell_ids) == n_cells

    # Convert to CSC format for efficient gene (column) access.
    X = X_cells_by_genes.tocsc()

    with open(out_path, "w") as f:

        # Header: first column is reserved for gene names.
        f.write("\t" + "\t".join(cell_ids) + "\n")

        # Process a small block of genes at a time to keep memory usage low.
        for start in range(0, n_genes, chunk_genes):
            end = min(start + chunk_genes, n_genes)

            # Convert only the current gene block to dense.
            block = X[:, start:end].toarray()  # shape: (n_cells, block_size)

            # Write one row per gene.
            for j in range(end - start):
                gene = gene_names[start + j]
                expression = block[:, j]

                f.write(gene)
                f.write("\t")

                # Write expression values separated by tabs.
                np.savetxt(
                    f,
                    expression.reshape(1, -1),
                    fmt="%.0f",
                    delimiter="\t",
                )

    print(
        f"Exported CytoTRACE2 expression matrix "
        f"({n_genes} genes × {n_cells} cells) to:\n{out_path}"
    )