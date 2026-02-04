import random
import numpy as np
import torch
import os
from src.tokenizer.gene_tokenizer import GeneVocab
from typing import Optional, Dict
import numpy as np
import pandas as pd
from scipy import stats, sparse
import networkx as nx

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def pick_eval_score(agg_results, scheme):
    df = agg_results.to_pandas()

    if scheme in ["pearson_delta", "mse", "mae", "mse_delta"]:
        return float(df[scheme].iloc[0])

    if scheme == "reverse":
        return float(df["pr_auc"].iloc[0])

    if scheme == "forward":
        pear = float(df["pearson_delta"].iloc[0])
        mse  = float(df["mse_delta"].iloc[0])
        alpha = 0.05 
        return pear - alpha * mse

    if scheme == "de":
        keys = ["de_spearman_sig", "de_direction_match", "de_sig_genes_recall"]
        return float(df[keys].iloc[0].mean())

    if scheme == "composite":
        pear = float(df["pearson_delta"].iloc[0])
        mse  = float(df["mse_delta"].iloc[0])
        pra  = float(df["pr_auc"].iloc[0])
        de   = float(df[["de_spearman_sig","de_direction_match","de_sig_genes_recall"]].iloc[0].mean())
        return 0.4*pra + 0.3*de + 0.3*(pear - 0.05*mse)

    raise ValueError("unknown scheme")

def make_lognorm_poisson_noise(target_log, alpha=1.0, per_cell_L=None, eps=1e-8):
    """
    target_log:  log1p(normalized_counts)
    alpha:       noise intensity (0.3~1.0; smaller is more conservative)
    per_cell_L:  if specified (e.g. 1e4), normalize expected total per cell to L
    """
    base = torch.expm1(target_log)                     
    if per_cell_L is not None or per_cell_L == -1 :
        scale = per_cell_L / (base.sum(dim=1, keepdim=True) + eps)
        lam = (base * scale).clamp_min(1e-8)           
    else:
        lam = (alpha * base).clamp_min(1e-8)
    x0_counts = torch.poisson(lam)                     
    x0_log = torch.log1p(x0_counts)                    
    return x0_log


def save_checkpoint(model, optimizer, scheduler, iteration, eval_score, save_path, is_best=False):
    """save checkpoint"""
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'eval_score': eval_score,
    }
    
    checkpoint_path = os.path.join(save_path, f'checkpoint.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"save checkpoint to: {checkpoint_path}")
    
    # If this is the best model, save an extra copy
    if is_best:
        best_path = os.path.join(save_path, 'best_checkpoint.pt')
        torch.save(checkpoint, best_path)
        print(f"save best checkpoint: {best_path}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """load checkpoint"""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        iteration = checkpoint['iteration']
        eval_score = checkpoint.get('eval_score', float('-inf'))
        print(f"loading {checkpoint_path} checkpoint, iteration: {iteration}, eval_score: {eval_score}")
        return iteration, eval_score
    else:
        print(f"Checkpoint file not found: {checkpoint_path}")
        return 0, float('-inf')
    
    
def process_vocab(data_manager, config):
    vocab_path = os.path.join('src/tokenizer',config.data_name+'_'+str(config.n_top_genes)+'_highly_vocab.json')
    if os.path.exists(vocab_path):
        print('##### loading vocab from file #####')
        vocab = GeneVocab.from_file(vocab_path)
    else:
        print('##### building vocab #####')
        highly_gene = data_manager.adata.var[data_manager.adata_train.var['highly_variable']]
        vocab = GeneVocab(list(highly_gene.index), specials=['<pad>', '<cls>', '<mask>', 'control'])
        vocab.save_json(vocab_path)
        vocab = GeneVocab.from_file(vocab_path)
    return vocab

def perturbation_id_to_emb_id(perturbation_id, vocab, perturbation_dict=None):
    device = perturbation_id.device
    inverse_dict = {v: str(k) for k, v in perturbation_dict.items()}
    perturbation_name = [inverse_dict[int(p_id)] for p_id in perturbation_id.cpu().numpy()]
    emb_id = vocab.encode(perturbation_name)
    emb_id = torch.tensor(emb_id, dtype=torch.long, device=device)
    return emb_id

def emb_id_to_perturbation_id(perturbation_id, vocab, perturbation_dict=None):
    device = perturbation_id.device
    inverse_dict = {v: str(k) for k, v in perturbation_dict.items()}
    perturbation_name = [inverse_dict[int(p_id)] for p_id in perturbation_id.cpu().numpy()]
    emb_id = vocab.encode(perturbation_name)
    emb_id = torch.tensor(emb_id, dtype=torch.long, device=device)
    return emb_id


@torch.no_grad()
def approximate_with_two_perturbations(
    model,
    perturb_vec: torch.Tensor,                 # target perturbation vector, shape (d,) or (1,d) or (B,d)
    candidate_ids: Optional[torch.Tensor]=None,# If None and not 'crisper', use all ntoken as candidates
    topk: int = 64,                            # preselect top-k candidates by cosine similarity
    nonneg_renorm: bool = True,                # clamp weights to non-negative and normalize to sum=1
    return_reconstruction: bool = True,
    topn_output: int = 10,                     # number of top results to return
    mode: str = "pair",                        # "pair" = combinations of two, "single" = single candidate
):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # 1) Normalize target vector to shape (B, d)
    if perturb_vec.dim() == 1:
        perturb_vec = perturb_vec.unsqueeze(0)
    perturb_vec = perturb_vec.to(device=device, dtype=dtype)  # (B, d)
    B, d = perturb_vec.shape

    # 2) Build candidate embedding matrix E: (N, d)
    if model.perturbation_function == 'crisper':
        assert candidate_ids is not None, "In 'crisper' mode, candidate_ids (N, L) must be provided."
        # Encode and average over sequence length
        E = model.encoder(candidate_ids.to(device=device)).to(dtype=dtype).mean(dim=1)
        cand_index = torch.arange(E.size(0), device=device)
    else:
        if candidate_ids is None:
            all_ids = torch.arange(model.perturbation_embedder.embedding.num_embeddings, device=device)
            E = model.perturbation_embedder(all_ids).to(dtype=dtype)
            cand_index = all_ids
        else:
            if candidate_ids.dim() == 2 and candidate_ids.size(1) == 1:
                candidate_ids = candidate_ids.squeeze(1)
            E = model.perturbation_embedder(candidate_ids.to(device=device)).to(dtype=dtype)
            cand_index = candidate_ids.to(device)

    N = E.size(0)
    assert N >= 1, "Need at least 1 candidate."

    # Mode: single perturbation search
    if mode == "single":
        E_norm = torch.nn.functional.normalize(E, dim=-1)
        P_norm = torch.nn.functional.normalize(perturb_vec, dim=-1)
        sims = P_norm @ E_norm.t()   # cosine similarity (B, N)
        k = min(topn_output, N)
        topn_val, topn_idx = torch.topk(sims, k=k, dim=-1)
        return {
            "topn_ids": cand_index[topn_idx],   # (B, k)
            "topn_scores": topn_val,            # (B, k)
        }

    # Mode: pairwise search
    elif mode == "pair":
        assert N >= 2, "Pair mode requires at least 2 candidates."

        # Preselect top-k candidates by cosine similarity
        E_norm = torch.nn.functional.normalize(E, dim=-1)
        P_norm = torch.nn.functional.normalize(perturb_vec, dim=-1)
        sims = P_norm @ E_norm.t()   # (B, N)
        k = min(topk, N)
        _, topk_idx = torch.topk(sims, k=k, dim=-1)  # (B, k)

        all_pairs = []
        I2 = torch.eye(2, device=device, dtype=dtype)

        for b in range(B):
            cand = E[topk_idx[b]]    # (k, d)
            idxs = topk_idx[b]       # (k,)
            pair_results = []

            for i in range(k):
                e_i = cand[i]
                for j in range(i + 1, k):
                    e_j = cand[j]
                    # Stack into matrix A: (d, 2)
                    A = torch.stack([e_i, e_j], dim=1)
                    ATA = A.t() @ A
                    ATp = A.t() @ perturb_vec[b]
                    # Solve least squares with small regularization for stability
                    w = torch.linalg.solve(ATA + 1e-6 * I2, ATp)

                    if nonneg_renorm:
                        w = torch.clamp(w, min=0)
                        s = w.sum()
                        if s > 0:
                            w = w / s

                    recon = A @ w
                    err = torch.norm(perturb_vec[b] - recon, p=2)

                    pair_results.append((
                        err.item(),
                        cand_index[idxs[i]].item(),
                        cand_index[idxs[j]].item(),
                        w.detach().clone().cpu(),
                        recon.detach().clone().cpu() if return_reconstruction else None
                    ))

            # Sort by error and keep top N
            pair_results.sort(key=lambda x: x[0])
            pair_results = pair_results[:min(topn_output, len(pair_results))]
            all_pairs.append(pair_results)

        return {
            "topn_pairs": all_pairs,  # length B; each is a list of up to topn_output tuples
        }

    else:
        raise ValueError("mode must be 'pair' or 'single'")

@torch.no_grad() 
def get_perturbation_embedding(model, perturbation_id: torch.Tensor) -> torch.Tensor:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    if model.perturbation_function == 'crisper':
        # (B, L) -> (B, L, d) -> mean over L -> (B, d)
        emb = model.encoder(perturbation_id.to(device=device)).to(dtype=dtype)
        emb = emb.mean(dim=1)
    else:
        # (B,) or (B,1) -> (B, d)
        if perturbation_id.dim() == 2 and perturbation_id.size(1) == 1:
            perturbation_id = perturbation_id.squeeze(1)
        emb = model.perturbation_embedder(perturbation_id.to(device=device)).to(dtype=dtype)
    return emb  # (B, d_model)

def freeze_backbone_for_p(vf):
    for name, p in vf.named_parameters():
        if name.startswith('p_head.') or name == 'p_mask_embed':
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

def unfreeze_all(vf):
    for _, p in vf.named_parameters():
        p.requires_grad_(True)
        
def set_requires_grad_for_p_only(vf, p_only: str):
    if hasattr(vf, "module"):
        base_vf = vf.module
    else:
        base_vf = vf
    for name, p in base_vf.named_parameters():
        if p_only == 'predict_p':
            if name.startswith("p_head.") or name == "p_mask_embed":
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
        else:
            p.requires_grad_(True)
            
@torch.no_grad()       
def get_perturbation_emb(vf, perturbation_id=None, perturbation_emb=None,
                            cell_1=None, use_mask: bool=False):
    if use_mask:
        B = cell_1.size(0)
        return vf.p_mask_embed[None, :].expand(B, -1).to(cell_1.device, dtype=cell_1.dtype)

    assert perturbation_emb is None or perturbation_id is None
    if perturbation_id is not None:
        if vf.perturbation_function == 'crisper':
            perturbation_emb = vf.encoder(perturbation_id)
        else:
            perturbation_emb = vf.perturbation_embedder(perturbation_id)
        perturbation_emb = perturbation_emb.mean(1)  # (B,d)
    elif perturbation_emb is not None:
        perturbation_emb = perturbation_emb.to(cell_1.device, dtype=cell_1.dtype)
        if perturbation_emb.dim() == 1:
            perturbation_emb = perturbation_emb.unsqueeze(0)
        if perturbation_emb.size(0) == 1:
            perturbation_emb = perturbation_emb.expand(cell_1.shape[0], -1).contiguous()
        perturbation_emb = vf.perturbation_embedder.enc_norm(perturbation_emb)
    return perturbation_emb


import numpy as np
import pandas as pd
from scipy import stats, sparse
import networkx as nx
import torch
from scipy import sparse


def preprocess_expression(X, log1p=False, zscore_per_gene=True):

    if isinstance(X, pd.DataFrame):
        X = X.values
    X = X.astype(np.float64, copy=False)
    if log1p:
        X = np.log1p(np.clip(X, a_min=0, a_max=None))
    if zscore_per_gene:
        # z-score per gene (column)
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, ddof=1, keepdims=True)
        std[std == 0] = 1.0
        X = (X - mean) / std
    return X

# ============ 1) Correlation matrix calculation ============
def correlation_matrix(X, method="pearson"):
    """
    X: (cells, genes)
    return: (genes, genes) correlation matrix in [-1, 1]
    """
    if method == "pearson":
        # np.corrcoef computes correlation by row by default, so transpose to compute by column (gene)
        C = np.corrcoef(X.T)
    elif method == "spearman":
        # Rank transform each gene, then Pearson
        X_rank = np.apply_along_axis(stats.rankdata, 0, X)
        C = np.corrcoef(X_rank.T)
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")
    # Numerical stability
    np.fill_diagonal(C, 1.0)
    C = np.clip(C, -1.0, 1.0)
    return C
def safe_correlation_matrix(X, method="pearson"):
    """
    X: (cells, genes). Returns (genes, genes) correlation matrix, no NaN/Inf.
    Rules:
    - If any gene has variance 0, set correlation with any other gene to 0 (self-correlation to 1)
    - If data contains NaN/Inf, replace with 0 (you can change to median/mean)
    """
    X = np.asarray(X, dtype=np.float64)

    # 1) Clean NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 2) Spearman: rank transform per column
    if method == "spearman":
        X = np.apply_along_axis(stats.rankdata, 0, X)

    n, g = X.shape

    # 3) Center
    Xc = X - X.mean(axis=0, keepdims=True)

    # 4) Column std (ddof=1), mark zero-variance columns
    std = Xc.std(axis=0, ddof=1)
    zero_var = std == 0.0

    # 5) Covariance matrix
    cov = (Xc.T @ Xc) / max(n - 1, 1)

    # 6) Build denominator and safe division
    denom = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        C = cov / denom

    # 7) Handle zero variance: wherever denominator is 0, set to 0; self-correlation to 1
    C[np.isnan(C)] = 0.0
    C[np.isinf(C)] = 0.0
    C[zero_var, :] = 0.0
    C[:, zero_var] = 0.0
    np.fill_diagonal(C, 1.0)

    # Clip
    np.clip(C, -1.0, 1.0, out=C)
    return C
# ============ 2) WGCNA-style soft threshold (optional) ============
def soft_threshold_weights(C, beta=6, use_abs=True):
    """
    C: correlation matrix (genes, genes)
    return: weighted adjacency (non-negative)
    """
    W = np.abs(C) if use_abs else np.maximum(C, 0.0)
    W = np.power(W, beta)
    np.fill_diagonal(W, 0.0)
    return W

# ============ 3A) Sparsification: Top-k ============
def sparsify_topk(W, k=10, keep_symmetry=True):
    """
    For each gene, keep k edges with largest absolute value.
    W: dense (genes, genes)
    return: csr sparse adjacency
    """
    n = W.shape[0]
    rows, cols, data = [], [], []
    for i in range(n):
        # Remove self-loop
        row = W[i].copy()
        row[i] = 0.0
        if k >= n - 1:
            idx = np.where(row != 0)[0]
        else:
            idx = np.argpartition(-np.abs(row), kth=min(k, n-2))[:k]
        vals = row[idx]
        mask = vals != 0
        rows.extend([i]*mask.sum())
        cols.extend(idx[mask])
        data.extend(vals[mask])

    A = sparse.csr_matrix((data, (rows, cols)), shape=W.shape)
    if keep_symmetry:
        # Take symmetric max or sum; here take max
        A = A.maximum(A.T)
    return A

# ============ 3B) Sparsification: threshold ============
def sparsify_threshold(W, tau=0.3, keep_symmetry=True):
    """
    Only keep edges where |W_ij| >= tau
    W: dense (genes, genes)
    """
    M = np.abs(W) >= tau
    np.fill_diagonal(M, False)
    rows, cols = np.where(M)
    data = W[rows, cols]
    A = sparse.csr_matrix((data, (rows, cols)), shape=W.shape)
    if keep_symmetry:
        A = A.maximum(A.T)
    return A

# ============ 4) GCN normalization ============
def gcn_normalize(A, add_self_loops=True):
    """
    Kipf & Welling GCN symmetric normalization: \hat A = D^{-1/2} (A + I) D^{-1/2}
    A: csr adjacency
    """
    if add_self_loops:
        A = A + sparse.eye(A.shape[0], format='csr')
    deg = np.array(A.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0
    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(deg))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return A_hat.tocsr()

# ============ 5) Build graph (optional, for visualization/export) ============
def to_networkx(A, gene_names=None, weight_attr="weight"):
    """
    A: csr adjacency
    """
    # Compatible with networkx versions
    if hasattr(nx, "from_scipy_sparse_array"):
        G = nx.from_scipy_sparse_array(A, edge_attribute=weight_attr)
    else:
        G = nx.from_scipy_sparse_matrix(A, edge_attribute=weight_attr)
        
    if gene_names is not None:
        mapping = {i: name for i, name in enumerate(gene_names)}
        G = nx.relabel_nodes(G, mapping)
    return G


def adjacency_to_mha_mask(A_csr: sparse.csr_matrix, allow_self=True):
    # Sparse adjacency -> dense 0/1 allowed matrix
    A = A_csr.tolil(copy=True)
    if allow_self:
        A.setdiag(1)
    A = A.tocsr().toarray().astype(bool)  # True=allowed edge
    disallow = ~A                         # True=disallowed (PyTorch bool mask semantics)
    # For MultiheadAttention, attn_mask shape can be (L, S)
    attn_mask_bool = torch.from_numpy(disallow)
    return attn_mask_bool  # True means masked

# ============ 6) One-stop pipeline ============
def build_gene_coexpression_graph(
    X,
    method="pearson",
    wgcna_beta=None,          # int or None; if given, use soft threshold weights
    sparsify="topk",          # "topk" or "threshold"
    k=10,                     # top-k
    tau=0.3,                  # threshold
    log1p=True,
    zscore_per_gene=True,
    use_negative_edge=False
):
    Xp = preprocess_expression(X, log1p=log1p, zscore_per_gene=zscore_per_gene)
    # C = correlation_matrix(Xp, method=method)
    C = safe_correlation_matrix(Xp, method=method)
    # Choose weight matrix source: use correlation directly or WGCNA style
    
    
    if wgcna_beta is not None:
        W = soft_threshold_weights(C, beta=wgcna_beta, use_abs=True)  
    else:
        W = C  

    sign_matrix = np.sign(W)
    if use_negative_edge:
        W = np.abs(W)
    # Sparsification
    if sparsify == "topk":
        A = sparsify_topk(W, k=k, keep_symmetry=True)
    elif sparsify == "threshold":
        A = sparsify_threshold(W, tau=tau, keep_symmetry=True)
    else:
        raise ValueError("sparsify must be 'topk' or 'threshold'")
    mask = adjacency_to_mha_mask(A)
    # return A, A_hat, G, sign_matrix
    return mask

def sorted_pad_mask(mask, pad_size=4, gene_names=None):
    sorted_gene_names = sorted(list(gene_names))
    reorder_idx = [gene_names.index(g) for g in sorted_gene_names]
    mask_sorted = mask[reorder_idx][:, reorder_idx]
    orig_shape = mask_sorted.shape
    pad_row = torch.ones((pad_size, orig_shape[1]), dtype=mask_sorted.dtype)
    pad_col = torch.ones((orig_shape[0] + pad_size, pad_size), dtype=mask_sorted.dtype)
    mask_padded = torch.cat([pad_row, mask_sorted], dim=0)
    mask_padded = torch.cat([pad_col, mask_padded], dim=1)
    length = mask.shape[0]
    mask_padded[torch.arange(length), torch.arange(length)] = False
    return mask_padded