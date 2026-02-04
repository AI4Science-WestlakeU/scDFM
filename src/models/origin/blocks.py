import pdb
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Tuple
import math
from .rotary import apply_rotary_emb

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'
    
    
class MultiheadDiffAttn_Origin(nn.Module):
    def __init__(
        self,
        embed_dim,
        depth, # current layer index
        num_heads,
        num_kv_heads=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # arg num_heads set to half of baseline Transformer's num_heads
        # for e.g., to compare with a baseline Transformer with 16 heads, pass in num_heads=8 for DIFF Transformer
        self.num_heads = num_heads
        
        # arg num_kv_heads set to half of baseline Transformer's num_kv_heads if use GQA
        # for e.g., to compare with a baseline Transformer with 16 heads and 8 kv_heads, 
        # pass in num_heads=8, num_kv_heads=4 for DIFF Transformer
        # if use MHA, pass in num_kv_heads=None
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        
        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # depth means current layer index
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)
    
    def forward(
        self,
        x,
        rel_pos,
        attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)

        q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        offset = src_len - tgt_len
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)
        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
        
        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)

        attn = self.out_proj(attn)
        return attn
    
class MultiheadDiffAttn(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        depth, # current layer index
        cross=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.cross = cross
        # The code `pdb.set_trace()` is setting a breakpoint in the code using the Python debugger
        # (pdb). When the code is executed, it will pause the program's execution at that point and
        # allow you to interactively debug the program.
        # pdb.set_trace()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj_1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj_1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj_2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj_2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # depth means current layer index
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(
        self,
        noisy_y,
        x,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len
        
        if self.cross:
            q_1 = self.q_proj_1(noisy_y)
            k_1 = self.k_proj_1(x)
            q_2 = self.q_proj_2(noisy_y)
            k_2 = self.k_proj_2(x)
        else:
            q_1 = self.q_proj_1(noisy_y)
            k_1 = self.k_proj_1(noisy_y)
            q_2 = self.q_proj_2(x)
            k_2 = self.k_proj_2(x)
        v = self.v_proj(noisy_y)

        q_1 = q_1.view(bsz, tgt_len, self.num_heads, self.head_dim)
        k_1 = k_1.view(bsz, src_len, self.num_heads, self.head_dim)
        q_2 = q_2.view(bsz, tgt_len, self.num_heads, self.head_dim)
        k_2 = k_2.view(bsz, src_len, self.num_heads, self.head_dim)

        v = v.view(bsz, src_len, self.num_heads, self.head_dim)


        q_1 = q_1.transpose(1, 2)
        q_2 = q_2.transpose(1, 2)
        k_1 = k_1.transpose(1, 2)
        k_2 = k_2.transpose(1, 2)
        q_1 *= self.scaling
        q_2 *= self.scaling

        attn_weights_1 = torch.matmul(q_1, k_1.transpose(-1, -2))
        attn_weights_2 = torch.matmul(q_2, k_2.transpose(-1, -2))

        attn_weights_1 = torch.nn.functional.softmax(attn_weights_1, dim=-1, dtype=torch.float32).type_as(
            attn_weights_1
        )  
        attn_weights_2 = torch.nn.functional.softmax(attn_weights_2, dim=-1, dtype=torch.float32).type_as(
            attn_weights_2
        )

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q_1)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q_1)

        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights_1 - lambda_full * attn_weights_2


        attn = torch.matmul(attn_weights, v.transpose(1,2))
        # attn: (bsz, num_heads, tgt_len, head_dim)        
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * self.head_dim)

        attn = self.out_proj(attn)
        return attn  



class CrossAttentionTransformerLayer(nn.Module):
    """
    One-layer Transformer block with ONLY cross-attention:
      - Pre-norm -> Multihead cross-attention (Q=tgt, K/V=memory)
      - Residual
      - Pre-norm -> FeedForward (MLP)
      - Residual

    Shapes (batch_first=True):
      tgt:    (B, T_tgt, d_model)
      memory: (B, T_mem, d_model)
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        batch_first: bool = True
    ):
        super().__init__()
        self.batch_first = batch_first

        # Cross attention: Q from tgt, K/V from memory
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=batch_first
        )
        self.cross_attn_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Feedforward
        self.linear1 = nn.Linear(d_model, int(mlp_ratio * d_model))
        self.linear2 = nn.Linear(int(mlp_ratio * d_model), d_model)
        self.ffn_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        average_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            tgt: (B, T_tgt, d_model)
            memory: (B, T_mem, d_model)
            attn_mask: optional mask applied to (tgt_len, mem_len) or (B*nhead, tgt_len, mem_len)
            memory_key_padding_mask: (B, T_mem) with True for positions to mask
            need_weights: return attention weights
            average_attn_weights: average heads if True

        Returns:
            out: (B, T_tgt, d_model)
            attn: (B, T_tgt, T_mem) or (B, nhead, T_tgt, T_mem) if need_weights else None
        """
        # --- Cross Attention (pre-norm) ---
        x = self.norm1(tgt)
        attn_out, attn_w = self.cross_attn(
            query=x,
            key=memory,
            value=memory,
            attn_mask=attn_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
        )
        x = tgt + self.cross_attn_dropout(attn_out)

        # --- Feedforward (pre-norm) ---
        y = self.norm2(x)
        y = self.linear2(self.ffn_dropout(self.activation(self.linear1(y))))
        out = x + self.ffn_dropout(y)

        return out, attn_w if need_weights else None
# class PerceiverDiffTransBlock(nn.Module):
#     def __init__(self, d_in, d_latent, heads=8, mlp_ratio=4):
#         super().__init__()

#         self.ln_z1 = nn.LayerNorm(d_latent)
#         self.q = nn.Linear(d_latent, d_latent)
#         self.k = nn.Linear(d_in, d_latent)
#         self.v = nn.Linear(d_in, d_latent)
        
#         self.q2 = nn.Linear(d_latent, d_latent)
#         self.k2 = nn.Linear(d_latent, d_latent)
#         self.v2 = nn.Linear(d_latent, d_latent)
#         self.cross = nn.MultiheadAttention(d_latent, heads, dropout=0.1, batch_first=True)

#         self.ln_z2 = nn.LayerNorm(d_latent)
#         self.self_attn = nn.MultiheadAttention(d_latent, heads, dropout=0.1, batch_first=True)
#         self.ln_z3 = nn.LayerNorm(d_latent)
#         self.mlp = nn.Sequential(
#             nn.Linear(d_latent, int(mlp_ratio * d_latent)), nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(int(mlp_ratio * d_latent), d_latent),
#             nn.Dropout(0.1),
#         )
#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(d_latent, 6 * d_latent, bias=True)
#         )
    
        
#     def forward(self, z, x, t):
#         shift_self, scale_self, gate_self, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)
#         z = z + self.cross(self.q(self.ln_z1(z)),
#                            self.k(x), self.v(x))[0]

#         z = modulate(self.ln_z2(z), shift_self, scale_self)
#         z = z + gate_self.unsqueeze(1) * self.self_attn(self.q2(z), self.k2(z), self.v2(z))[0]

#         z = z + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.ln_z3(z), shift_mlp, scale_mlp))
#         return z