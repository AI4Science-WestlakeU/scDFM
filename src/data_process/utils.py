import anndata as ad
import pandas as pd
import numpy as np

def build_generated_anndata(
    embeddings: np.ndarray,          # shape: (N, D)
    condition_id: np.ndarray,        # shape: (N, 2)
    pert_col: str = "perturbation",  # first perturbation column name
    pert_col2: str = "perturbation_2",  # second perturbation (if combo)
    combo_col: str = "pert_combo",   # combined label
) -> ad.AnnData:
    """
    Construct AnnData from model-generated embeddings and condition IDs.
    """
    N = embeddings.shape[0]
    if condition_id.shape[0] != N:
        raise ValueError("condition_id and embeddings must have same first dimension.")

    # obs: per-cell annotation
    obs = pd.DataFrame({
        pert_col: condition_id[:, 0],
        pert_col2: condition_id[:, 1],
        combo_col: [f"{a}_{b}" for a, b in condition_id],
    }, index=[f"cell_{i}" for i in range(N)])

    # construct AnnData
    adata = ad.AnnData(
        X=embeddings,  # could also store in obsm
        obs=obs
    )

    # also store embedding explicitly for plotting
    adata.obsm["embedding"] = embeddings

    return adata