import torch.nn as nn
from typing import Optional
from torch import nn, Tensor
import pdb
class CategoryValueEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.long()
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x
    
class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x

class PerturbationEmbedding(nn.Module):
    def __init__(self, num_perturbations, emb_dim, max_comb_len=2, fusion_method='mlp', output_matrix=False):
        super().__init__()
        self.embedding = nn.Embedding(num_perturbations, emb_dim)
        self.fusion_method = fusion_method
        self.max_comb_len = max_comb_len
        self.output_matrix = output_matrix
        self.output_dim = emb_dim if not output_matrix else emb_dim * emb_dim

        if fusion_method == 'mlp':
            self.fusion = nn.Sequential(
                nn.Linear(emb_dim * max_comb_len, emb_dim * 2),
                nn.ReLU(),
                nn.Linear(emb_dim * 2, self.output_dim)
            )
        elif fusion_method == 'sum':
            self.fusion = None
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")

    def forward(self, ids):
        emb = self.embedding(ids)  # [B, C, D]
        
        if self.fusion_method == 'mlp':
            emb = emb.view(emb.size(0), -1)  # [B, C*D]
            fused = self.fusion(emb)         # [B, D] or [B, D*D]

            if self.output_matrix:
                B = fused.size(0)
                D = int(self.output_dim ** 0.5)
                return fused.view(B, D, D)   # [B, D, D]
            else:
                return fused

        elif self.fusion_method == 'sum':
            out = emb.sum(dim=1)  # [B, D]
            if self.output_matrix:
                B = out.size(0)
                D = out.size(1)
                return out.view(B, D, 1).expand(B, D, D)  # dummy expansion
            return out
        
# class PerturbationEmbedding(nn.Module):
#     def __init__(self, num_perturbations, emb_dim, max_comb_len=2, fusion_method='mlp'):
#         """
#         Args:
#             num_perturbations: 词表大小
#             emb_dim: 嵌入维度
#             max_comb_len: 每个 condition 最多包含的 token 数量（如 drug1, drug2）
#             fusion_method: 'mlp' 或 'sum'
#         """
#         super().__init__()
#         self.embedding = nn.Embedding(num_perturbations, emb_dim)
#         self.fusion_method = fusion_method
#         self.max_comb_len = max_comb_len

#         if fusion_method == 'mlp':
#             self.fusion = nn.Sequential(
#                 nn.Linear(emb_dim * max_comb_len, emb_dim),
#                 nn.ReLU(),
#                 nn.Linear(emb_dim, emb_dim)
#             )
#         elif fusion_method == 'sum':
#             self.fusion = None
#         else:
#             raise ValueError(f"Unsupported fusion method: {fusion_method}")
#     def init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             nn.init.zeros_(m.bias)
            
#     def initialize_weights(self):
#         self.apply(self.init_weights)
        
#     def forward(self, ids):
#         """
#         Args:
#             ids: LongTensor of shape [B, max_comb_len]
#         Returns:
#             fused: Tensor of shape [B, emb_dim]
#         """
        
#         emb = self.embedding(ids)  # [B, C, D]
        
#         if self.fusion_method == 'mlp':
#             emb = emb.view(emb.size(0), -1)  # [B, C*D]
#             return self.fusion(emb)          # [B, D]

#         elif self.fusion_method == 'sum':
#             if emb.dim() == 2:
#                 return emb.sum(dim=0)            # [B, D]
#             elif emb.dim() == 3:
#                 return emb.sum(dim=1)            # [B, C, D]
#             else:
#                 raise ValueError(f"Unsupported dimension: {ids.dim()}")