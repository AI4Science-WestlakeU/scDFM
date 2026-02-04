import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, Mapping, Optional, Tuple, Any, Union
import pdb
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from src.flow_matching.utils.model_wrapper import ModelWrapper

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class PerturbationEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train=True, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class CrossAttentionModule(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          batch_first=True)  # using (batch, seq, dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x1, x2, attn_mask=None, key_padding_mask=None):
        # x1: (b, l1, dim), x2: (b, l2, dim)
        residual = x1
        attn_out, attn_weights = self.attn(query=x1, key=x2, value=x2,
                                           attn_mask=attn_mask,
                                           key_padding_mask=key_padding_mask)
        x = self.norm(attn_out + residual)

        # add FFN
        residual2 = x
        x = self.ff(x)
        x = self.norm(x + residual2)

        return x, attn_weights
    

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    
class HierarchicalBlock(nn.Module):
    def __init__(self,d_model,nhead,dropout):
        super().__init__()
        self.cell1_block = DiTBlock(d_model,nhead,dropout)
        self.cell2_block = DiTBlock(d_model,nhead,dropout)
        self.cross_attention = CrossAttentionModule(d_model,nhead,dropout)
        
    def forward(self,cell1,cell2,perturbation,t):
        cell1 = self.cell1_block(cell1,perturbation+t)
        cell2 = self.cell2_block(cell2,t)
        cell1, _ = self.cross_attention(cell2,cell1)
        
        return cell1,cell2,
        
class SymmetricBlock(nn.Module):
    def __init__(self,d_model,nhead,dropout):
        super().__init__()
        # self.hierarchical_block1 = HierarchicalBlock(d_model,nhead,dropout)
        # self.hierarchical_block2 = HierarchicalBlock(d_model,nhead,dropout)
        self.hierarchical_block = HierarchicalBlock(d_model,nhead,dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self,cell1,cell2,perturbation,t):
        # cell1_1,cell2_1 = self.hierarchical_block1(cell1,cell2,perturbation,t)
        # cell2_2,cell1_2 = self.hierarchical_block2(cell2,cell1,-perturbation,t)
        cell1_1,cell2_1 = self.hierarchical_block(cell1,cell2,perturbation,t)
        cell2_2,cell1_2 = self.hierarchical_block(cell2,cell1,-perturbation,t)
        cell1 = self.norm(cell1_1 + cell1_2)
        cell2 = self.norm(cell2_1 + cell2_2)
        return cell1,cell2
        
class Model(nn.Module):
    def __init__(
        self,
        model_type: str = 'hierarchical',
        ntoken: int = 512,
        d_model: int = 512,
        nhead: int = 8,
        nlayers: int = 6 ,
        n_combination: int = 2,
        nlayers_cls: int = 3,
        d_perturbation: int = 512,
        vocab: Any = None,
        dropout: float = 0.5, 
    ):
        super().__init__()
        self.n_combination = n_combination
        self.d_model = d_model
        self.model_type = model_type
        self.ntoken = ntoken
        self.d_perturbation = d_perturbation
        # self.p_embedder = PerturbationEmbedder(n_perturbation, d_model, dropout_prob=0)
        self.p_embedder = nn.Sequential(
            nn.Linear(d_perturbation,d_model),
            nn.SiLU(),
            nn.Linear(d_model,d_model),
        )
        self.t_embedder = TimestepEmbedder(d_model)
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, nhead, dropout) for _ in range(nlayers)
        ])

        if model_type == 'hierarchical':
            self.blocks = nn.ModuleList([
                SymmetricBlock(d_model, nhead, dropout) for _ in range(nlayers)
            ])
        else:
            self.blocks = nn.ModuleList([
                DiTBlock(d_model, nhead, dropout) for _ in range(nlayers)
            ])
        self.perturbation_decoder = nn.Sequential(
            nn.Linear(d_model,d_model*4),
            nn.SiLU(),
            nn.Linear(d_model*4,d_model),
            )
            
            
    def forward(self,cell1,cell2,t,perturbation=None):
        """_summary_

        Args:
            cell1 (tensor): (B,ntoken,d_model) for control cell
            cell2 (tensor): (B,ntoken,d_model) for perturbed cell
            t (long): (B,) when inference is a number
            perturbation (long): (B,n_combination)
        """
        t = self.t_embedder(t)                   # (N, D)
        # pdb.set_trace()
        if perturbation is not None:
            perturbation_ = perturbation.reshape(-1,self.d_perturbation)
            p = self.p_embedder(perturbation_)        # (N, D)
            # p = p.reshape(-1,self.n_combination,self.d_model)
            # p = p.sum(dim=1)
            c = t + p
        else:
            c = t
        cells = torch.cat([cell1,cell2],dim=1)
        for block in self.blocks:
            if self.model_type == 'hierarchical':
                cell1,cell2 = block(cell1,cell2,p,t)
            else:
                cells = block(cells,c)
        if self.model_type != 'hierarchical':
            cells = cells.reshape(-1,2,self.ntoken,self.d_model)
            cell1 = cells[:,0,:,:]  
            cell2 = cells[:,1,:,:]
        p = self.perturbation_decoder((cell1 + cell2).mean(dim=1))
        # p = p.reshape(-1,self.n_combination)
        
        return cell1, cell2, p



class TimedTransformer(nn.Module):
    def __init__(
        self,
        model_type: str = 'hierarchical',
        ntoken: int = 512,
        d_model: int = 512,
        nhead: int = 8,
        nlayers: int = 8 ,
        n_combination: int = 2,
        nlayers_cls: int = 3,
        d_perturbation: int = 512,
        vocab: Any = None,
        dropout: float = 0.5, 
    ):
        super().__init__()
        self.n_combination = n_combination
        self.d_model = d_model
        self.model_type = model_type
        self.ntoken = ntoken
        # self.p_embedder = PerturbationEmbedder(n_perturbation, d_model, dropout_prob=0)
        self.t_embedder = TimestepEmbedder(d_model)
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, nhead, dropout) for _ in range(nlayers)
        ])

            
            
    def forward(self,cell1,t):
        """_summary_

        Args:
            cell1 (tensor): (B,ntoken,d_model) for control cell
            cell2 (tensor): (B,ntoken,d_model) for perturbed cell
            t (long): (B,) when inference is a number
            perturbation (long): (B,n_combination)
        """
        t = self.t_embedder(t)                   # (N, D)
        c = t
        cells = cell1
        for block in self.blocks:
            cells = block(cells,c)
        return cells
        

# class WrappedModel(ModelWrapper):
#     def __init__(self, model, mode="noisy_y"):
#         super().__init__(model)
#         assert mode in ["noisy_y", "noisy_p"]
#         self.mode = mode

#     def forward(self, x: torch.Tensor, t: torch.Tensor, x_t: torch.Tensor, condition_vec: torch.Tensor, mode=None):
#         """
#         x: x_0
#         x_t: path sample at time t
#         t: time step
#         condition_vec: condition vector
#         """
#         if mode is not None:
#             self.mode = mode
#         if self.mode == "noisy_y":
#             _, predicted_x_t, _ = self.model(x, x_t, t, condition_vec)
#             return (x, predicted_x_t, condition_vec)
#         elif self.mode == "noisy_p":
#             _, _, predicted_p_t = self.model(x, x_t, t, condition_vec)
#             return (x, x_t, predicted_p_t)
#         else:
#             raise ValueError(f"Unsupported mode: {self.mode}")

class WrappedModel(ModelWrapper):
    def __init__(self, model, mode="noisy_y"):
        super().__init__(model)
        assert mode in ["noisy_y", "noisy_p"]
        self.mode = mode
        self.model = model

    def change_mode(self, mode):
        self.mode = mode

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        if t.ndim == 0:
            t = t.expand(x.shape[0]).to(x.device)

        if self.mode == "noisy_y":
            condition_vec = extras["condition_vec"]
            x_0 = extras["x_0"]
            _, predicted_x_t, _ = self.model(x_0, x, t, condition_vec)
            return predicted_x_t
        elif self.mode == "noisy_p":
            target_y = extras["target_y"]
            x_0 = extras["x_0"]
            _, _, predicted_p_t = self.model(x_0, target_y, t, x)
            return predicted_p_t
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

if __name__ == '__main__':
    batch_size = 128
    model = Model(model_type='hs',ntoken=512).cuda()
    cell1 = torch.randn(batch_size, 1024,512).cuda()
    cell2 = torch.randn(batch_size, 1024,512).cuda()
    t = torch.randint(0, 1000, (batch_size,)).cuda()
    # p = torch.randint(0,2,(5,2)).cuda()
    p = torch.randn(batch_size,512).cuda()
    model = nn.DataParallel(model,device_ids=[0,1,2,3,4,5,6,7])
    
    from tqdm import tqdm   
    for i in tqdm(range(5000)):
        out = model(cell1,cell2, t,p)
        out[0].sum().backward()
    pdb.set_trace()
    print(out.shape)
    
    # from accelerate import Accelerator
    # import time

    # accelerator = Accelerator()
    # batch_size = 128
    # model = Model(model_type='hs', ntoken=512)
    # cell1 = torch.randn(batch_size, 1024, 512)
    # cell2 = torch.randn(batch_size, 1024, 512)
    # t = torch.randint(0, 1000, (batch_size,))
    # p = torch.randn(batch_size, 512)

    # # 用 accelerator.prepare 包装
    # model, cell1, cell2, t, p = accelerator.prepare(model, cell1, cell2, t, p)

    # from tqdm import tqdm
    # start = time.time()
    # for i in tqdm(range(5000)):
    #     out = model(cell1, cell2, t, p)
    #     loss = out[0].sum() + out[1].sum() + out[2].sum()
    #     accelerator.backward(loss)
    # end = time.time()
    # print(f"Total time: {end - start:.2f} seconds")
    # import pdb; pdb.set_trace()
    # print(out[0].shape)