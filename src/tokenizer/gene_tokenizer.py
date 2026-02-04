import json
import pickle
from pathlib import Path
from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple, Union
from typing_extensions import Self

import numpy as np
import pandas as pd
import torch

# Remove torchtext dependency and create custom Vocab class
class Vocab:
    """
    Custom vocabulary implementation to replace torchtext.vocab.Vocab
    """
    
    def __init__(self, token2idx: Dict[str, int]):
        self.stoi = token2idx
        self.itos = {idx: token for token, idx in token2idx.items()}
        self.default_index = None
        self.default_token = None
    
    def __getitem__(self, token: str) -> int:
        return self.stoi[token]
    
    def __contains__(self, token: str) -> bool:
        return token in self.stoi
    
    def __len__(self) -> int:
        return len(self.stoi)
    
    def get_stoi(self) -> Dict[str, int]:
        return self.stoi
    
    def get_itos(self) -> Dict[int, str]:
        return self.itos
    
    def insert_token(self, token: str, index: int) -> None:
        """Insert a token at a specific index"""
        self.stoi[token] = index
        self.itos[index] = token
    
    def set_default_index(self, index: int) -> None:
        """Set the default index for unknown tokens"""
        self.default_index = index
    
    def set_default_token(self, token: str) -> None:
        """Set the default token"""
        if token in self.stoi:
            self.default_token = token
            self.default_index = self.stoi[token]
    
    def encode(self, tokens: Union[str, List[str], List[List[str]]]) -> Union[int, List[int], List[List[int]]]:
        """
        Convert tokens to indices.
        
        Args:
            tokens: Single token (str) or list of tokens
            
        Returns:
            Single index (int) or list of indices
        """
        if isinstance(tokens, str):
            return self.stoi[tokens]
        elif isinstance(tokens, list) and not isinstance(tokens[0], list):
            return [self.stoi[token] for token in tokens]
        elif isinstance(tokens, list) and isinstance(tokens[0], list):
            return [[self.stoi[token] for token in token_list] for token_list in tokens]
        else:
            raise TypeError(f"Expected str or list of str, got {type(tokens)}")
    
    def decode(self, indices: Union[int, List[int], torch.Tensor, np.ndarray]) -> Union[str, List[str], torch.Tensor, np.ndarray]:
        """
        Convert indices to tokens.
        
        Args:
            indices: Single index (int) or list of indices
            
        Returns:
            Single token (str) or list of tokens
        """
        if isinstance(indices, int):
            return self.itos[indices]
        elif isinstance(indices, list):
            return [self.itos[idx] for idx in indices]
        elif isinstance(indices, torch.Tensor) or isinstance(indices, np.ndarray):
            if len(indices.shape) > 1:
                return [self.decode(indices[i]) for i in range(indices.shape[0])]
            else:
                return [self.itos[idx] for idx in indices]
        else:
            raise TypeError(f"Expected int or list of int, got {type(indices)}")
    
    def encode_batch(self, batch_tokens: List[List[str]]) -> List[List[int]]:
        """
        Convert a batch of token lists to indices.
        
        Args:
            batch_tokens: List of token lists
            
        Returns:
            List of index lists
        """
        return [self.encode(tokens) for tokens in batch_tokens]
    
    def decode_batch(self, batch_indices: List[List[int]]) -> List[List[str]]:
        """
        Convert a batch of index lists to tokens.
        
        Args:
            batch_indices: List of index lists
            
        Returns:
            List of token lists
        """
        return [self.decode(indices) for indices in batch_indices]
    
    def lookup_indices(self, tokens: List[str]) -> List[int]:
        """
        Look up indices for a list of tokens, handling unknown tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of indices, using default_index for unknown tokens
        """
        indices = []
        for token in tokens:
            if token in self.stoi:
                indices.append(self.stoi[token])
            elif self.default_index is not None:
                indices.append(self.default_index)
            else:
                raise KeyError(f"Token '{token}' not found in vocabulary and no default index set")
        return indices
    
    def lookup_tokens(self, indices: List[int]) -> List[str]:
        """
        Look up tokens for a list of indices.
        
        Args:
            indices: List of indices
            
        Returns:
            List of tokens
        """
        tokens = []
        for idx in indices:
            if idx in self.itos:
                tokens.append(self.itos[idx])
            else:
                raise KeyError(f"Index {idx} not found in vocabulary")
        return tokens


class GeneVocab(Vocab):
    """
    Vocabulary for genes.
    """

    def __init__(
        self,
        gene_list_or_vocab: Union[List[str], Vocab],
        specials: Optional[List[str]] = None,
        special_first: bool = True,
        default_token: Optional[str] = "<pad>",
    ) -> None:
        """
        Initialize the vocabulary.
        Note: add specials only works when init from a gene list.

        Args:
            gene_list_or_vocab (List[str] or Vocab): List of gene names or a
                Vocab object.
            specials (List[str]): List of special tokens.
            special_first (bool): Whether to add special tokens to the beginning
                of the vocabulary.
            default_token (str): Default token, by default will set to "<pad>",
                if "<pad>" is in the vocabulary.
        """
        if isinstance(gene_list_or_vocab, Vocab):
            _vocab = gene_list_or_vocab
            if specials is not None:
                raise ValueError(
                    "receive non-empty specials when init from a Vocab object."
                )
        elif isinstance(gene_list_or_vocab, list):
            _vocab = self._build_vocab_from_iterator(
                gene_list_or_vocab,
                specials=specials,
                special_first=special_first,
            )
        else:
            raise ValueError(
                "gene_list_or_vocab must be a list of gene names or a Vocab object."
            )
        super().__init__(_vocab.stoi)
        if default_token is not None and default_token in self:
            self.set_default_token(default_token)

    @classmethod
    def from_file(cls, file_path: Union[Path, str]) -> Self:
        """
        Load the vocabulary from a file. The file should be either a pickle or a
        json file of token to index mapping.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if file_path.suffix == ".pkl":
            with file_path.open("rb") as f:
                vocab = pickle.load(f)
                return cls(vocab)
        elif file_path.suffix == ".json":
            with file_path.open("r") as f:
                token2idx = json.load(f)
                return cls.from_dict(token2idx)
        else:
            raise ValueError(
                f"{file_path} is not a valid file type. "
                "Only .pkl and .json are supported."
            )

    @classmethod
    def from_dict(
        cls,
        token2idx: Dict[str, int],
        default_token: Optional[str] = "<pad>",
    ) -> Self:
        """
        Load the vocabulary from a dictionary.

        Args:
            token2idx (Dict[str, int]): Dictionary mapping tokens to indices.
        """
        # initiate an empty vocabulary first
        _vocab = cls([])

        # add the tokens to the vocabulary, GeneVocab requires consecutive indices
        for t, i in sorted(token2idx.items(), key=lambda x: x[1]):
            _vocab.insert_token(t, i)

        if default_token is not None and default_token in _vocab:
            _vocab.set_default_token(default_token)

        return _vocab

    def _build_vocab_from_iterator(
        self,
        iterator: Iterable,
        min_freq: int = 1,
        specials: Optional[List[str]] = None,
        special_first: bool = True,
    ) -> Vocab:
        """
        Build a Vocab from an iterator. This function is modified from
        torchtext.vocab.build_vocab_from_iterator. The original function always
        splits tokens into characters, which is not what we want.

        Args:
            iterator (Iterable): Iterator used to build Vocab. Must yield list
                or iterator of tokens.
            min_freq (int): The minimum frequency needed to include a token in
                the vocabulary.
            specials (List[str]): Special symbols to add. The order of supplied
                tokens will be preserved.
            special_first (bool): Whether to add special tokens to the beginning

        Returns:
            Vocab: A `Vocab` object
        """

        counter = Counter()
        counter.update(iterator)

        if specials is not None:
            for tok in specials:
                del counter[tok]

        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[0])
        sorted_by_freq_tuples.sort(key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)

        if specials is not None:
            if special_first:
                specials = specials[::-1]
            for symbol in specials:
                ordered_dict.update({symbol: min_freq})
                ordered_dict.move_to_end(symbol, last=not special_first)

        # Create token2idx mapping
        token2idx = {}
        for i, (token, _) in enumerate(ordered_dict.items()):
            if ordered_dict[token] >= min_freq:
                token2idx[token] = i
        
        return Vocab(token2idx)

    @property
    def pad_token(self) -> Optional[str]:
        """
        Get the pad token.
        """
        if getattr(self, "_pad_token", None) is None:
            self._pad_token = None
        return self._pad_token

    @pad_token.setter
    def pad_token(self, pad_token: str) -> None:
        """
        Set the pad token. Will not add the pad token to the vocabulary.

        Args:
            pad_token (str): Pad token, should be in the vocabulary.
        """
        if pad_token not in self:
            raise ValueError(f"{pad_token} is not in the vocabulary.")
        self._pad_token = pad_token

    def save_json(self, file_path: Union[Path, str]) -> None:
        """
        Save the vocabulary to a json file.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        with file_path.open("w") as f:
            json.dump(self.get_stoi(), f, indent=2)

    def set_default_token(self, default_token: str) -> None:
        """
        Set the default token.

        Args:
            default_token (str): Default token.
        """
        if default_token not in self:
            raise ValueError(f"{default_token} is not in the vocabulary.")
        self.set_default_index(self[default_token])
    
    def get_gene_list(self) -> List[str]:
        """
        Get the list of all genes in the vocabulary.
        
        Returns:
            List of gene names
        """
        return list(self.stoi.keys())
    
    def get_gene_indices(self) -> List[int]:
        """
        Get the list of all gene indices in the vocabulary.
        
        Returns:
            List of gene indices
        """
        return list(self.itos.keys())
    
    def add_gene(self, gene_name: str) -> int:
        """
        Add a new gene to the vocabulary.
        
        Args:
            gene_name: Name of the gene to add
            
        Returns:
            Index assigned to the gene
        """
        if gene_name in self.stoi:
            return self.stoi[gene_name]
        
        new_index = len(self.stoi)
        self.insert_token(gene_name, new_index)
        return new_index
    
    def add_genes(self, gene_names: List[str]) -> List[int]:
        """
        Add multiple genes to the vocabulary.
        
        Args:
            gene_names: List of gene names to add
            
        Returns:
            List of indices assigned to the genes
        """
        indices = []
        for gene_name in gene_names:
            indices.append(self.add_gene(gene_name))
        return indices
    
    def filter_genes(self, gene_names: List[str]) -> 'GeneVocab':
        """
        Create a new vocabulary containing only the specified genes.
        
        Args:
            gene_names: List of gene names to keep
            
        Returns:
            New GeneVocab with filtered genes
        """
        filtered_stoi = {gene: self.stoi[gene] for gene in gene_names if gene in self.stoi}
        return GeneVocab.from_dict(filtered_stoi, default_token=self.default_token)
    
    def get_gene_frequency(self, gene_name: str) -> int:
        """
        Get the frequency of a gene in the vocabulary.
        Note: This is a placeholder method since we don't track frequencies in this implementation.
        
        Args:
            gene_name: Name of the gene
            
        Returns:
            Frequency of the gene (always 1 in this implementation)
        """
        if gene_name in self.stoi:
            return 1
        return 0


def get_default_gene_vocab() -> GeneVocab:
    """
    Get the default gene vocabulary, consisting of gene symbols and ids.
    """
    vocab_file = Path(__file__).parent / "default_gene_vocab.json"
    if not vocab_file.exists():
        print(
            f"No existing default vocab, will build one and save to {vocab_file}"
        )
        return _build_default_gene_vocab(save_vocab_to=vocab_file)
    print(f"Loading gene vocabulary from {vocab_file}")
    return GeneVocab.from_file(vocab_file)


def _build_default_gene_vocab(
    download_source_to: str = "/tmp",
    save_vocab_to: Union[Path, str, None] = None,
) -> GeneVocab:
    """
    Build the default gene vocabulary from HGNC gene symbols.

    Args:
        download_source_to (str): Directory to download the source data.
        save_vocab_to (Path or str): Path to save the vocabulary. If None,
            the vocabulary will not be saved. Default to None.
    """
    gene_collection_file = (
        Path(download_source_to) / "human.gene_name_symbol.from_genenames.org.tsv"
    )
    if not gene_collection_file.exists():
        # download and save file from url
        url = (
            "https://www.genenames.org/cgi-bin/download/custom?col=gd_app_sym&"
            "col=md_ensembl_id&status=Approved&status=Entry%20Withdrawn&hgnc_dbtag"
            "=on&order_by=gd_app_sym_sort&format=text&submit=submit"
        )
        import requests

        r = requests.get(url)
        gene_collection_file.write_text(r.text)

    print(f"Building gene vocabulary from {gene_collection_file}")
    df = pd.read_csv(gene_collection_file, sep="\t")
    gene_list = df["Approved symbol"].dropna().unique().tolist()
    gene_vocab = GeneVocab(gene_list)  # no special tokens set in default vocab
    if save_vocab_to is not None:
        gene_vocab.save_json(Path(save_vocab_to))
    return gene_vocab


def tokenize_batch(
    data: np.ndarray,
    gene_ids: np.ndarray,
    return_pt: bool = True,
    append_cls: bool = True,
    include_zero_gene: bool = False,
    cls_id: int = "<cls>",
    mod_type: np.ndarray = None,
    cls_id_mod_type: int = None,
) -> List[Tuple[Union[torch.Tensor, np.ndarray]]]:
    """
    Tokenize a batch of data. Returns a list of tuple (gene_id, count).

    Args:
        data (array-like): A batch of data, with shape (batch_size, n_features).
            n_features equals the number of all genes.
        gene_ids (array-like): A batch of gene ids, with shape (n_features,).
        return_pt (bool): Whether to return torch tensors of gene_ids and counts,
            default to True.

    Returns:
        list: A list of tuple (gene_id, count) of non zero gene expressions.
    """
    if data.shape[1] != len(gene_ids):
        raise ValueError(
            f"Number of features in data ({data.shape[1]}) does not match "
            f"number of gene_ids ({len(gene_ids)})."
        )
    if mod_type is not None and data.shape[1] != len(mod_type):
        raise ValueError(
            f"Number of features in data ({data.shape[1]}) does not match "
            f"number of mod_type ({len(mod_type)})."
        )

    tokenized_data = []
    for i in range(len(data)):
        row = data[i]
        mod_types = None
        if include_zero_gene:
            values = row
            genes = gene_ids
            if mod_type is not None:
                mod_types = mod_type
        else:
            idx = np.nonzero(row)[0]
            values = row[idx]
            genes = gene_ids[idx]
            if mod_type is not None:
                mod_types = mod_type[idx]
        if append_cls:
            genes = np.insert(genes, 0, cls_id)
            values = np.insert(values, 0, 0)
            if mod_type is not None:
                mod_types = np.insert(mod_types, 0, cls_id_mod_type)
        if return_pt:
            genes = torch.from_numpy(genes).long()
            values = torch.from_numpy(values).float()
            if mod_type is not None:
                mod_types = torch.from_numpy(mod_types).long()
        tokenized_data.append((genes, values, mod_types))
    return tokenized_data


def pad_batch(
    batch: List[Tuple],
    max_len: int,
    vocab: Vocab,
    pad_token: str = "<pad>",
    pad_value: int = 0,
    cls_appended: bool = True,
    vocab_mod: Vocab = None,
) -> Dict[str, torch.Tensor]:
    """
    Pad a batch of data. Returns a list of Dict[gene_id, count].

    Args:
        batch (list): A list of tuple (gene_id, count).
        max_len (int): The maximum length of the batch.
        vocab (Vocab): The vocabulary containing the pad token.
        pad_token (str): The token to pad with.

    Returns:
        Dict[str, torch.Tensor]: A dictionary of gene_id and count.
    """
    max_ori_len = max(len(batch[i][0]) for i in range(len(batch)))
    max_len = min(max_ori_len, max_len)

    pad_id = vocab[pad_token]
    if vocab_mod is not None:
        mod_pad_id = vocab_mod[pad_token]
    gene_ids_list = []
    values_list = []
    mod_types_list = []

    for i in range(len(batch)):
        gene_ids, values, mod_types = batch[i]

        if len(gene_ids) > max_len:
            # sample max_len genes
            if not cls_appended:
                idx = np.random.choice(len(gene_ids), max_len, replace=False)
            else:
                idx = np.random.choice(len(gene_ids) - 1, max_len - 1, replace=False)
                idx = idx + 1
                idx = np.insert(idx, 0, 0)
            gene_ids = gene_ids[idx]
            values = values[idx]
            if mod_types is not None:
                mod_types = mod_types[idx]
        if len(gene_ids) < max_len:
            gene_ids = torch.cat(
                [
                    gene_ids,
                    torch.full(
                        (max_len - len(gene_ids),), pad_id, dtype=gene_ids.dtype
                    ),
                ]
            )
            values = torch.cat(
                [
                    values,
                    torch.full((max_len - len(values),), pad_value, dtype=values.dtype),
                ]
            )
            if mod_types is not None:
                mod_types = torch.cat(
                    [
                        mod_types,
                        torch.full(
                            (max_len - len(mod_types),),
                            mod_pad_id,
                            dtype=mod_types.dtype,
                        ),
                    ]
                )

        gene_ids_list.append(gene_ids)
        values_list.append(values)
        if mod_types is not None:
            mod_types_list.append(mod_types)

    batch_padded = {
        "genes": torch.stack(gene_ids_list, dim=0),
        "values": torch.stack(values_list, dim=0),
    }
    if mod_types is not None:
        batch_padded["mod_types"] = torch.stack(mod_types_list, dim=0)
    return batch_padded


def tokenize_and_pad_batch(
    data: np.ndarray,
    gene_ids: np.ndarray,
    max_len: int,
    vocab: Vocab,
    pad_token: str,
    pad_value: int,
    append_cls: bool = True,
    include_zero_gene: bool = False,
    cls_token: str = "<cls>",
    return_pt: bool = True,
    mod_type: np.ndarray = None,
    vocab_mod: Vocab = None,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize and pad a batch of data. Returns a list of tuple (gene_id, count).
    """
    cls_id = vocab[cls_token]
    if mod_type is not None:
        cls_id_mod_type = vocab_mod[cls_token]
    tokenized_data = tokenize_batch(
        data,
        gene_ids,
        return_pt=return_pt,
        append_cls=append_cls,
        include_zero_gene=include_zero_gene,
        cls_id=cls_id,
        mod_type=mod_type,
        cls_id_mod_type=cls_id_mod_type if mod_type is not None else None,
    )

    batch_padded = pad_batch(
        tokenized_data,
        max_len,
        vocab,
        pad_token,
        pad_value,
        cls_appended=append_cls,
        vocab_mod=vocab_mod,
    )
    return batch_padded


def random_mask_value(
    values: Union[torch.Tensor, np.ndarray],
    mask_ratio: float = 0.15,
    mask_value: int = -1,
    pad_value: int = 0,
    random_mask_zero_gene: bool = False,
) -> torch.Tensor:
    """
    Randomly mask a batch of data.

    Args:
        values (array-like):
            A batch of tokenized data, with shape (batch_size, n_features).
        mask_ratio (float): The ratio of genes to mask, default to 0.15.
        mask_value (int): The value to mask with, default to -1.
        pad_value (int): The value of padding in the values, will be kept unchanged.

    Returns:
        torch.Tensor: A tensor of masked data.
    """
    if isinstance(values, torch.Tensor):
        # it is crutial to clone the tensor, otherwise it changes the original tensor
        values = values.clone().detach().numpy()
    else:
        values = values.copy()

    for i in range(len(values)):
        row = values[i]
        non_padding_idx = np.nonzero(row - pad_value)[0]
        n_mask = int(len(non_padding_idx) * mask_ratio)
        mask_idx = np.random.choice(non_padding_idx, n_mask, replace=False)
        if random_mask_zero_gene:
            zero_idx = np.where(row == 0)[0]
            if len(zero_idx) > 0:
                n_mask_zero = min(n_mask, len(zero_idx))
                mask_zero_idx = np.random.choice(zero_idx, n_mask_zero, replace=False)
                row[mask_zero_idx] = mask_value
        row[mask_idx] = mask_value
    return torch.from_numpy(values).float()

def category_noise(values: Union[torch.Tensor, np.ndarray], mask_ratio: float = 0.15, pad_value: int = 0, n_bins: int = 200):
    """
    Randomly replace values with noise from [0, n_bins] range and return both noisy values and mask.
    
    Args:
        values (array-like): A batch of tokenized data, with shape (batch_size, n_features)
        mask_ratio (float): The ratio of values to replace with noise, default to 0.15
        pad_value (int): The value of padding in the values, will be kept unchanged
        n_bins (int): The maximum bin value (exclusive) for noise generation
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - noisy data tensor with same shape as input
            - binary mask tensor indicating which positions were noised (1 for noised, 0 for unchanged)
    """
    if isinstance(values, torch.Tensor):
        values = values.clone().detach().numpy()
    else:
        values = values.copy()
        
    # Initialize mask array with same shape as values
    mask = np.zeros_like(values)
        
    for i in range(len(values)):
        row = values[i]
        row_mask = mask[i]
        
        # Get indices of non-zero and non-padding values
        nonzero_idx = np.where((row != 0) & (row != pad_value))[0]
        zero_idx = np.where(row == 0)[0]
        
        # Calculate number of values to noise for each category
        n_noise_nonzero = int(len(nonzero_idx) * mask_ratio)
        n_noise_zero = int(len(zero_idx) * mask_ratio)
        # n_noise_zero = int(n_noise_nonzero/10)
        # Randomly select indices to noise
        if len(nonzero_idx) > 0:
            noise_nonzero_idx = np.random.choice(nonzero_idx, n_noise_nonzero, replace=False)
            row[noise_nonzero_idx] = np.random.randint(0, n_bins, size=n_noise_nonzero)
            row_mask[noise_nonzero_idx] = 1
            
        if len(zero_idx) > 0:
            noise_zero_idx = np.random.choice(zero_idx, n_noise_zero, replace=False)
            row[noise_zero_idx] = np.random.randint(0, n_bins, size=n_noise_zero)
            row_mask[noise_zero_idx] = 1
            
    return torch.from_numpy(values).float(), torch.from_numpy(mask).bool()

def batch_encode_genes(
    gene_lists: List[List[str]], 
    vocab: Vocab,
    max_length: Optional[int] = None,
    pad_token: str = "<pad>",
    truncate: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Encode a batch of gene lists to indices with padding.
    
    Args:
        gene_lists: List of gene name lists
        vocab: Vocabulary to use for encoding
        max_length: Maximum length for padding/truncation
        pad_token: Token to use for padding
        truncate: Whether to truncate sequences longer than max_length
        
    Returns:
        Dictionary with 'gene_ids' tensor and 'attention_mask' tensor
    """
    if max_length is None:
        max_length = max(len(genes) for genes in gene_lists)
    
    batch_size = len(gene_lists)
    gene_ids = []
    attention_mask = []
    
    for genes in gene_lists:
        # Encode genes to indices
        indices = vocab.encode(genes)
        
        # Truncate if necessary
        if truncate and len(indices) > max_length:
            indices = indices[:max_length]
        
        # Pad if necessary
        if len(indices) < max_length:
            pad_length = max_length - len(indices)
            indices.extend([vocab[pad_token]] * pad_length)
            mask = [1] * len(genes) + [0] * pad_length
        else:
            mask = [1] * max_length
        
        gene_ids.append(indices)
        attention_mask.append(mask)
    
    return {
        'gene_ids': torch.tensor(gene_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
    }


def batch_decode_genes(
    gene_ids: torch.Tensor,
    vocab: Vocab,
    attention_mask: Optional[torch.Tensor] = None,
    pad_token: str = "<pad>"
) -> List[List[str]]:
    """
    Decode a batch of gene indices back to gene names.
    
    Args:
        gene_ids: Tensor of shape (batch_size, seq_length) containing gene indices
        vocab: Vocabulary to use for decoding
        attention_mask: Optional attention mask to ignore padding tokens
        pad_token: Token to ignore during decoding
        
    Returns:
        List of gene name lists
    """
    batch_size = gene_ids.shape[0]
    decoded_genes = []
    
    for i in range(batch_size):
        indices = gene_ids[i].tolist()
        genes = []
        
        for j, idx in enumerate(indices):
            # Skip padding tokens
            if attention_mask is not None and attention_mask[i][j] == 0:
                continue
            if idx == vocab[pad_token]:
                continue
                
            gene_name = vocab.decode(idx)
            genes.append(gene_name)
        
        decoded_genes.append(genes)
    
    return decoded_genes


def create_gene_vocab_from_data(
    gene_data: List[List[str]],
    min_freq: int = 1,
    special_tokens: Optional[List[str]] = None,
    default_token: str = "<pad>"
) -> GeneVocab:
    """
    Create a gene vocabulary from gene data.
    
    Args:
        gene_data: List of gene lists
        min_freq: Minimum frequency for a gene to be included
        special_tokens: Special tokens to add to vocabulary
        default_token: Default token for unknown genes
        
    Returns:
        GeneVocab object
    """
    # Flatten all gene lists
    all_genes = []
    for gene_list in gene_data:
        all_genes.extend(gene_list)
    
    # Count frequencies
    gene_counts = Counter(all_genes)
    
    # Filter by minimum frequency
    filtered_genes = [gene for gene, count in gene_counts.items() if count >= min_freq]
    
    # Add special tokens if provided
    if special_tokens:
        filtered_genes = special_tokens + filtered_genes
    
    return GeneVocab(filtered_genes, specials=special_tokens, default_token=default_token)


def merge_gene_vocabs(
    vocab1: GeneVocab,
    vocab2: GeneVocab,
    default_token: str = "<pad>"
) -> GeneVocab:
    """
    Merge two gene vocabularies.
    
    Args:
        vocab1: First vocabulary
        vocab2: Second vocabulary
        default_token: Default token for the merged vocabulary
        
    Returns:
        Merged GeneVocab object
    """
    # Get all genes from both vocabularies
    all_genes = list(vocab1.get_gene_list()) + list(vocab2.get_gene_list())
    
    # Remove duplicates while preserving order
    unique_genes = []
    seen = set()
    for gene in all_genes:
        if gene not in seen:
            unique_genes.append(gene)
            seen.add(gene)
    
    return GeneVocab(unique_genes, default_token=default_token)

if __name__ == "__main__":

    vocab = GeneVocab(['GENE1', 'GENE2', 'GENE3'], specials=['<pad>', '<cls>'])

    indices = vocab.encode(['GENE1', 'GENE2'])
    genes = vocab.decode(indices)

    gene_lists = [['GENE1', 'GENE2'], ['GENE3']]
    batch_result = batch_encode_genes(gene_lists, vocab, max_length=5)

    gene_data = [['GENE1', 'GENE2'], ['GENE2', 'GENE3']]
    vocab_from_data = create_gene_vocab_from_data(gene_data, min_freq=2)