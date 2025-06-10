# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import List, Optional

import torch
from torch import nn
import numpy as np

from torchtune.models.llama3._model_utils import scale_hidden_dim_for_mlp
from torchtune.models.llama3_1._position_embeddings import Llama3ScaledRoPE
from torchtune.modules import TiedLinear
from torchtune.modules import (
    MultiHeadAttention,
    FeedForward,
    FrozenNF4Linear,
    RMSNorm,
    TransformerDecoder,
    TransformerSelfAttentionLayer,
)

from torchtune.modules.common_utils import reparametrize_as_dtype_state_dict_post_hook

from torchtune.modules.peft import DoRALinear, LORA_ATTN_MODULES, LoRALinear

from PIL import Image

"""
Component builders for the Llama3.2 model and popular variants such as LoRA.

torchtune provides composable building blocks. Builder functions help
stitch these building blocks into higher-level components. This design has
two benefits:
- The building blocks themselves are very flexible. For example, ``MultiHeadAttention``
can take either nn.Linear or nn.LoRALinear for ``q_proj``.
- Builder functions expose a set of configurable params which keep the constructors of
the building blocks simple.
"""


# ------------------ Vanilla Llama3.2 ------------------


def llama3_2_clevr3d(
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    rope_base: int = 500_000,
    intermediate_dim: Optional[int] = None,
    norm_eps: float = 1e-5,
    scale_factor: int = 32,
    added_tokens_offset: int = 128256,
    vqgan_embed_dim: int = 256,
    vqgan_start_index: int = 129471,
    vqgan_end_index: int = 137663,
    vqgan_vocab_size: int = 8192,
    vqgan_codebook_path: str = "./3d-mllm-datasets-and-codebooks/vqgan-models-and-codebooks/domain-agnostic/quantize_weight_8192.npy",
    image_token_offset: int = 129471,
    use_vqgan_codebook: bool = True,
    use_sin_cos_numbers: bool = False,
    sin_cos_numbers_offset: int = 1201,
    use_mlp_projector: bool = False,
    use_sin_cos_plus_learned: bool = False,
    no_independent_numbers: bool = False,
    no_independent_numbers_no_3d_tokens: bool = False,
) -> TransformerDecoder:
    """
    Build the decoder associated with the Llama3.2 model. This includes:
    - Token embeddings
    - num_layers number of TransformerSelfAttentionLayer blocks
    - RMS Norm layer applied to the output of the transformer
    - Final projection into token space

    Args:
        vocab_size (int): number of tokens in vocabulary.
        num_layers (int): number of layers in the transformer decoder.
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. User should ensure
            `num_heads` % `num_kv_heads` == 0. For standard MHA set `num_kv_heads` == `num_heads`,
            for GQA `num_kv_heads` < `num_heads`, and for MQA set `num_kv_heads` == 1.
        embed_dim (int): embedding dimension for self-attention
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`
        rope_base (int): base for the rotary positional embeddings. Default: 500_000
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        intermediate_dim (Optional[int]): intermediate dimension for MLP. If not specified,
            this is computed using :func:`~torchtune.modules.scale_hidden_dim_for_mlp`
        norm_eps (float): epsilon in RMS norms.
        scale_factor (int): scaling factor for RoPE. Default: 32

    Returns:
        TransformerDecoder: Instantiation of Llama3.2 model.
    """
    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    rope = Llama3ScaledRoPE(
        dim=head_dim, max_seq_len=max_seq_len, base=rope_base, scale_factor=scale_factor
    )
    layers = []
    for _ in range(num_layers):
        self_attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
            pos_embeddings=rope,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
        )
        hidden_dim = (
            intermediate_dim
            if intermediate_dim
            else scale_hidden_dim_for_mlp(embed_dim)
        )
        mlp = llama3_mlp(dim=embed_dim, hidden_dim=hidden_dim)
        layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
            mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        )
        layers.append(layer)
    layers = nn.ModuleList(layers)

    # tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    # output_proj = TiedLinear(tok_embeddings)
    # assert that both use_sin_cos_numbers and use_sin_cos_plus_learned are not True, else print an error
    assert not (
        use_sin_cos_numbers and use_sin_cos_plus_learned
    ), "Both use_sin_cos_numbers and use_sin_cos_plus_learned cannot be True"
    if use_sin_cos_numbers:
        tok_embeddings = CLEVRThreeDEmbeddingWithSinCosNumbers(
            vocab_size,
            embed_dim,
            added_tokens_offset,
            vqgan_embed_dim,
            vqgan_start_index,
            vqgan_end_index,
            vqgan_vocab_size,
            vqgan_codebook_path,
            image_token_offset,
            use_vqgan_codebook,
            sin_cos_numbers_offset,
            use_mlp_projector,
        )
    elif use_sin_cos_plus_learned:
        tok_embeddings = CLEVRThreeDEmbeddingWithSinCosNumbersPlusLearned(
            vocab_size,
            embed_dim,
            added_tokens_offset,
            vqgan_embed_dim,
            vqgan_start_index,
            vqgan_end_index,
            vqgan_vocab_size,
            vqgan_codebook_path,
            image_token_offset,
            use_vqgan_codebook,
            sin_cos_numbers_offset,
            use_mlp_projector,
        )
    elif no_independent_numbers:
        tok_embeddings = (
            CLEVRThreeDEmbeddingWithSinCosNumbersPlusLearnedNoIndependentNumbers(
                vocab_size,
                embed_dim,
                added_tokens_offset,
                vqgan_embed_dim,
                vqgan_start_index,
                vqgan_end_index,
                vqgan_vocab_size,
                vqgan_codebook_path,
                image_token_offset,
                use_vqgan_codebook,
                sin_cos_numbers_offset,
                use_mlp_projector,
            )
        )
    elif no_independent_numbers_no_3d_tokens:
        tok_embeddings = CLEVRThreeDEmbeddingWithSinCosNumbersPlusLearnedNoIndependentNumbersNo3DTokens(
            vocab_size,
            embed_dim,
            added_tokens_offset,
            vqgan_embed_dim,
            vqgan_start_index,
            vqgan_end_index,
            vqgan_vocab_size,
            vqgan_codebook_path,
            image_token_offset,
            use_vqgan_codebook,
            sin_cos_numbers_offset,
            use_mlp_projector,
        )
    else:
        tok_embeddings = CLEVRThreeDEmbedding(
            vocab_size,
            embed_dim,
            added_tokens_offset,
            vqgan_embed_dim,
            vqgan_start_index,
            vqgan_end_index,
            vqgan_vocab_size,
            vqgan_codebook_path,
            image_token_offset,
            use_vqgan_codebook,
        )
    output_proj = nn.Linear(
        embed_dim,
        vocab_size + (image_token_offset - added_tokens_offset) + vqgan_vocab_size,
        bias=False,
    )

    return TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
    )


def get_sin_cos_positional_embedding(num_positions, embedding_dim):
    """
    Generate sinusoidal positional embeddings.

    :param num_positions: Number of positions (sequence length)
    :param embedding_dim: Dimensionality of the embeddings
    :return: A matrix of shape (num_positions, embedding_dim) with positional embeddings
    """
    # Initialize the position indices and dimension indices
    position = np.arange(num_positions)[:, np.newaxis]  # Shape: (num_positions, 1)
    dim = np.arange(embedding_dim)[np.newaxis, :]  # Shape: (1, embedding_dim)

    # Compute the term for scaling position and dimension
    angle_rates = 1 / np.power(10000, (2 * (dim // 2)) / np.float32(embedding_dim))

    # Apply sin to even indices and cos to odd indices
    positional_embedding = np.zeros((num_positions, embedding_dim))
    positional_embedding[:, 0::2] = np.sin(
        position * angle_rates[:, 0::2]
    )  # Sin on even indices
    positional_embedding[:, 1::2] = np.cos(
        position * angle_rates[:, 1::2]
    )  # Cos on odd indices

    return positional_embedding


# custom embedding class with sin-cos positional embeddings for numbers
class CLEVRThreeDEmbeddingWithSinCosNumbers(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        added_tokens_offset: int,
        vqgan_embed_dim: int,
        vqgan_start_index: int,
        vqgan_end_index: int,
        vqgan_vocab_size: int,
        vqgan_codebook_path: str,
        image_token_offset: int,
        use_vqgan_codebook: bool,
        sin_cos_numbers_offset: int,
        use_mlp_projector: bool = False,
    ):
        super().__init__()
        self.sin_cos_numbers_offset = sin_cos_numbers_offset
        self.use_mlp_projector = use_mlp_projector
        self.embed_dim = embed_dim
        self.added_tokens_offset = added_tokens_offset
        self.vqgan_embed_dim = vqgan_embed_dim
        self.vqgan_start_index = vqgan_start_index
        self.vqgan_end_index = vqgan_end_index
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.added_embedding = nn.Embedding(
            image_token_offset - added_tokens_offset - self.sin_cos_numbers_offset,
            embed_dim,
        )
        #######
        self.numbers_embedding = nn.Embedding(self.sin_cos_numbers_offset, embed_dim)
        sin_cos_values = get_sin_cos_positional_embedding(
            self.sin_cos_numbers_offset, embed_dim
        )
        self.numbers_embedding.weight.data = torch.tensor(
            sin_cos_values, dtype=torch.bfloat16
        )
        self.numbers_embedding.weight.requires_grad = False
        #######
        self.use_vqgan_codebook = use_vqgan_codebook
        self.vqgan_codebook = nn.Embedding(vqgan_vocab_size, vqgan_embed_dim)
        if self.use_vqgan_codebook:
            if self.use_mlp_projector:
                # two layer MLP projector with ReLU activation
                self.vqgan_embed_proj = nn.Sequential(
                    nn.Linear(vqgan_embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, embed_dim),
                )
            else:
                self.vqgan_embed_proj = nn.Linear(
                    vqgan_embed_dim, embed_dim, bias=False
                )
        else:
            self.vqgan_embed_proj = nn.Embedding(vqgan_vocab_size, embed_dim)
        # load the quantized weights from the path which points to a npy file
        self.vqgan_codebook.weight.data = torch.tensor(
            np.load(vqgan_codebook_path), dtype=torch.bfloat16
        )
        # make requires_grad False
        self.vqgan_codebook.weight.requires_grad = False
        self.image_token_offset = image_token_offset

    def forward(self, x):
        # x is a tensor of shape (batch_size, seq_len) containing token ids
        # separate the token ids into text and image token ids
        # text token ids are values in the range 0 to vqgan_start_index
        # image token ids are values in the range vqgan_start_index to vqgan_end_index

        text_mask = (x >= 0) & (x < self.added_tokens_offset)
        number_mask = (x >= self.added_tokens_offset) & (
            x < self.added_tokens_offset + self.sin_cos_numbers_offset
        )
        three_d_mask = (x >= self.added_tokens_offset + self.sin_cos_numbers_offset) & (
            x < self.vqgan_start_index
        )
        image_mask = (x >= self.vqgan_start_index) & (x < self.vqgan_end_index)

        embeddings = torch.zeros(
            x.shape[0],
            x.shape[1],
            self.embed_dim,
            device=x.device,
            dtype=torch.bfloat16,
        )

        # Initialize tensors for text and image tokens
        text_tokens = torch.zeros_like(x)
        numbers_tokens = torch.zeros_like(x)
        three_d_tokens = torch.zeros_like(x)
        image_tokens = torch.zeros_like(x)

        # Fill the tensors with the corresponding token ids
        text_tokens[text_mask] = x[text_mask]
        numbers_tokens[number_mask] = x[number_mask]
        three_d_tokens[three_d_mask] = x[three_d_mask]
        image_tokens[image_mask] = x[image_mask]

        image_tokens -= self.image_token_offset
        # make negative values 0
        image_tokens[image_tokens < 0] = 0

        numbers_tokens -= self.added_tokens_offset
        # make negative values 0
        numbers_tokens[numbers_tokens < 0] = 0

        three_d_tokens -= self.added_tokens_offset + self.sin_cos_numbers_offset
        # make negative values 0
        three_d_tokens[three_d_tokens < 0] = 0

        # get the embeddings for the text and image tokens considering the text and image masks
        text_embeddings = self.token_embedding(text_tokens)

        if self.use_vqgan_codebook:
            image_embeddings = self.vqgan_codebook(image_tokens)
            image_embeddings = self.vqgan_embed_proj(image_embeddings)
        else:
            image_embeddings = self.vqgan_embed_proj(image_tokens)

        numbers_embeddings = self.numbers_embedding(numbers_tokens)
        three_d_embeddings = self.added_embedding(three_d_tokens)

        embeddings[text_mask] = text_embeddings[text_mask]
        embeddings[image_mask] = image_embeddings[image_mask]
        embeddings[number_mask] = numbers_embeddings[number_mask]
        embeddings[three_d_mask] = three_d_embeddings[three_d_mask]

        return embeddings


# custom embedding class with sin-cos positional embeddings added to learned embeddings for numbers
class CLEVRThreeDEmbeddingWithSinCosNumbersPlusLearned(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        added_tokens_offset: int,
        vqgan_embed_dim: int,
        vqgan_start_index: int,
        vqgan_end_index: int,
        vqgan_vocab_size: int,
        vqgan_codebook_path: str,
        image_token_offset: int,
        use_vqgan_codebook: bool,
        sin_cos_numbers_offset: int,
        use_mlp_projector: bool = False,
    ):
        super().__init__()
        self.sin_cos_numbers_offset = sin_cos_numbers_offset
        self.use_mlp_projector = use_mlp_projector
        self.embed_dim = embed_dim
        self.added_tokens_offset = added_tokens_offset
        self.vqgan_embed_dim = vqgan_embed_dim
        self.vqgan_start_index = vqgan_start_index
        self.vqgan_end_index = vqgan_end_index
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.added_embedding = nn.Embedding(
            image_token_offset - added_tokens_offset - self.sin_cos_numbers_offset,
            embed_dim,
        )
        #######
        self.numbers_embedding = nn.Embedding(self.sin_cos_numbers_offset, embed_dim)
        sin_cos_values = get_sin_cos_positional_embedding(
            self.sin_cos_numbers_offset, embed_dim
        )
        # self.numbers_embedding.weight.data = torch.tensor(
        # sin_cos_values, dtype=torch.bfloat16
        # )
        # self.numbers_embedding.weight.requires_grad = False
        self.static_sin_cos_embedding = nn.Embedding(
            self.sin_cos_numbers_offset, embed_dim
        )
        self.static_sin_cos_embedding.weight.data = torch.tensor(
            sin_cos_values, dtype=torch.bfloat16
        )
        self.static_sin_cos_embedding.weight.requires_grad = False
        #######
        self.use_vqgan_codebook = use_vqgan_codebook
        self.vqgan_codebook = nn.Embedding(vqgan_vocab_size, vqgan_embed_dim)
        if self.use_vqgan_codebook:
            if self.use_mlp_projector:
                # two layer MLP projector with ReLU activation
                self.vqgan_embed_proj = nn.Sequential(
                    nn.Linear(vqgan_embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, embed_dim),
                )
            else:
                self.vqgan_embed_proj = nn.Linear(
                    vqgan_embed_dim, embed_dim, bias=False
                )
        else:
            self.vqgan_embed_proj = nn.Embedding(vqgan_vocab_size, embed_dim)
        # load the quantized weights from the path which points to a npy file
        self.vqgan_codebook.weight.data = torch.tensor(
            np.load(vqgan_codebook_path), dtype=torch.bfloat16
        )
        # make requires_grad False
        self.vqgan_codebook.weight.requires_grad = False
        self.image_token_offset = image_token_offset

    def forward(self, x):
        # x is a tensor of shape (batch_size, seq_len) containing token ids
        # separate the token ids into text and image token ids
        # text token ids are values in the range 0 to vqgan_start_index
        # image token ids are values in the range vqgan_start_index to vqgan_end_index

        text_mask = (x >= 0) & (x < self.added_tokens_offset)
        number_mask = (x >= self.added_tokens_offset) & (
            x < self.added_tokens_offset + self.sin_cos_numbers_offset
        )
        three_d_mask = (x >= self.added_tokens_offset + self.sin_cos_numbers_offset) & (
            x < self.vqgan_start_index
        )
        image_mask = (x >= self.vqgan_start_index) & (x < self.vqgan_end_index)

        embeddings = torch.zeros(
            x.shape[0],
            x.shape[1],
            self.embed_dim,
            device=x.device,
            dtype=torch.bfloat16,
        )

        # Initialize tensors for text and image tokens
        text_tokens = torch.zeros_like(x)
        numbers_tokens = torch.zeros_like(x)
        three_d_tokens = torch.zeros_like(x)
        image_tokens = torch.zeros_like(x)

        # Fill the tensors with the corresponding token ids
        text_tokens[text_mask] = x[text_mask]
        numbers_tokens[number_mask] = x[number_mask]
        three_d_tokens[three_d_mask] = x[three_d_mask]
        image_tokens[image_mask] = x[image_mask]

        image_tokens -= self.image_token_offset
        # make negative values 0
        image_tokens[image_tokens < 0] = 0

        numbers_tokens -= self.added_tokens_offset
        # make negative values 0
        numbers_tokens[numbers_tokens < 0] = 0

        three_d_tokens -= self.added_tokens_offset + self.sin_cos_numbers_offset
        # make negative values 0
        three_d_tokens[three_d_tokens < 0] = 0

        # get the embeddings for the text and image tokens considering the text and image masks
        text_embeddings = self.token_embedding(text_tokens)

        if self.use_vqgan_codebook:
            image_embeddings = self.vqgan_codebook(image_tokens)
            image_embeddings = self.vqgan_embed_proj(image_embeddings)
        else:
            image_embeddings = self.vqgan_embed_proj(image_tokens)

        numbers_embeddings = self.numbers_embedding(
            numbers_tokens
        ) + self.static_sin_cos_embedding(numbers_tokens)
        three_d_embeddings = self.added_embedding(three_d_tokens)

        embeddings[text_mask] = text_embeddings[text_mask]
        embeddings[image_mask] = image_embeddings[image_mask]
        embeddings[number_mask] = numbers_embeddings[number_mask]
        embeddings[three_d_mask] = three_d_embeddings[three_d_mask]

        return embeddings


# no independent numbers
class CLEVRThreeDEmbeddingWithSinCosNumbersPlusLearnedNoIndependentNumbers(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        added_tokens_offset: int,
        vqgan_embed_dim: int,
        vqgan_start_index: int,
        vqgan_end_index: int,
        vqgan_vocab_size: int,
        vqgan_codebook_path: str,
        image_token_offset: int,
        use_vqgan_codebook: bool,
        sin_cos_numbers_offset: int,
        use_mlp_projector: bool = False,
    ):
        super().__init__()
        self.sin_cos_numbers_offset = sin_cos_numbers_offset
        self.use_mlp_projector = use_mlp_projector
        self.embed_dim = embed_dim
        self.added_tokens_offset = added_tokens_offset
        self.vqgan_embed_dim = vqgan_embed_dim
        self.vqgan_start_index = vqgan_start_index
        self.vqgan_end_index = vqgan_end_index
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.added_embedding = nn.Embedding(
            image_token_offset - added_tokens_offset - self.sin_cos_numbers_offset,
            embed_dim,
        )
        #######
        # self.numbers_embedding = nn.Embedding(
        #     self.sin_cos_numbers_offset, embed_dim
        # )  # TODO: remove hardcoding
        # sin_cos_values = get_sin_cos_positional_embedding(
        #     self.sin_cos_numbers_offset, embed_dim
        # )
        # self.numbers_embedding.weight.data = torch.tensor(
        # sin_cos_values, dtype=torch.bfloat16
        # )
        # self.numbers_embedding.weight.requires_grad = False
        # self.static_sin_cos_embedding = nn.Embedding(
        #     self.sin_cos_numbers_offset, embed_dim
        # )
        # self.static_sin_cos_embedding.weight.data = torch.tensor(
        #     sin_cos_values, dtype=torch.bfloat16
        # )
        # self.static_sin_cos_embedding.weight.requires_grad = False
        #######
        self.use_vqgan_codebook = use_vqgan_codebook
        self.vqgan_codebook = nn.Embedding(vqgan_vocab_size, vqgan_embed_dim)
        if self.use_vqgan_codebook:
            if self.use_mlp_projector:
                # two layer MLP projector with ReLU activation
                self.vqgan_embed_proj = nn.Sequential(
                    nn.Linear(vqgan_embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, embed_dim),
                )
            else:
                self.vqgan_embed_proj = nn.Linear(
                    vqgan_embed_dim, embed_dim, bias=False
                )
        else:
            self.vqgan_embed_proj = nn.Embedding(vqgan_vocab_size, embed_dim)
        # load the quantized weights from the path which points to a npy file
        self.vqgan_codebook.weight.data = torch.tensor(
            np.load(vqgan_codebook_path), dtype=torch.bfloat16
        )
        # make requires_grad False
        self.vqgan_codebook.weight.requires_grad = False
        self.image_token_offset = image_token_offset

    def forward(self, x):
        # x is a tensor of shape (batch_size, seq_len) containing token ids
        # separate the token ids into text and image token ids
        # text token ids are values in the range 0 to vqgan_start_index
        # image token ids are values in the range vqgan_start_index to vqgan_end_index
        # make sure to consider the shape of x and the batch size
        text_mask = (x >= 0) & (x < self.added_tokens_offset)
        # number_mask = (x >= self.added_tokens_offset) & (
        #     x < self.added_tokens_offset + self.sin_cos_numbers_offset
        # )
        three_d_mask = (x >= self.added_tokens_offset + self.sin_cos_numbers_offset) & (
            x < self.vqgan_start_index
        )
        image_mask = (x >= self.vqgan_start_index) & (x < self.vqgan_end_index)

        # assert the sum of masks is equal to the total number of tokens
        assert (
            text_mask.sum() + three_d_mask.sum() + image_mask.sum() == x.numel()
        ), "Sum of masks is not equal to the total number of tokens"

        embeddings = torch.zeros(
            x.shape[0],
            x.shape[1],
            self.embed_dim,
            device=x.device,
            dtype=torch.bfloat16,
        )

        # Initialize tensors for text and image tokens
        text_tokens = torch.zeros_like(x)
        # numbers_tokens = torch.zeros_like(x)
        three_d_tokens = torch.zeros_like(x)
        image_tokens = torch.zeros_like(x)

        # Fill the tensors with the corresponding token ids
        text_tokens[text_mask] = x[text_mask]
        # numbers_tokens[number_mask] = x[number_mask]
        three_d_tokens[three_d_mask] = x[three_d_mask]
        image_tokens[image_mask] = x[image_mask]

        image_tokens -= self.image_token_offset
        # make negative values 0
        image_tokens[image_tokens < 0] = 0

        # numbers_tokens -= self.added_tokens_offset
        # make negative values 0
        # numbers_tokens[numbers_tokens < 0] = 0

        three_d_tokens -= self.added_tokens_offset + self.sin_cos_numbers_offset
        # make negative values 0
        three_d_tokens[three_d_tokens < 0] = 0

        # get the embeddings for the text and image tokens considering the text and image masks
        text_embeddings = self.token_embedding(text_tokens)

        if self.use_vqgan_codebook:
            image_embeddings = self.vqgan_codebook(image_tokens)
            image_embeddings = self.vqgan_embed_proj(image_embeddings)
        else:
            image_embeddings = self.vqgan_embed_proj(image_tokens)

        # numbers_embeddings = self.numbers_embedding(
        #     numbers_tokens
        # ) + self.static_sin_cos_embedding(numbers_tokens)
        three_d_embeddings = self.added_embedding(three_d_tokens)

        embeddings[text_mask] = text_embeddings[text_mask]
        embeddings[image_mask] = image_embeddings[image_mask]
        # embeddings[number_mask] = numbers_embeddings[number_mask]
        embeddings[three_d_mask] = three_d_embeddings[three_d_mask]

        return embeddings


# no independent numbers no 3D tokens
class CLEVRThreeDEmbeddingWithSinCosNumbersPlusLearnedNoIndependentNumbersNo3DTokens(
    nn.Module
):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        added_tokens_offset: int,
        vqgan_embed_dim: int,
        vqgan_start_index: int,
        vqgan_end_index: int,
        vqgan_vocab_size: int,
        vqgan_codebook_path: str,
        image_token_offset: int,
        use_vqgan_codebook: bool,
        sin_cos_numbers_offset: int,
        use_mlp_projector: bool = False,
    ):
        super().__init__()
        self.sin_cos_numbers_offset = sin_cos_numbers_offset
        self.use_mlp_projector = use_mlp_projector
        self.embed_dim = embed_dim
        self.added_tokens_offset = added_tokens_offset
        self.vqgan_embed_dim = vqgan_embed_dim
        self.vqgan_start_index = vqgan_start_index
        self.vqgan_end_index = vqgan_end_index
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # self.added_embedding = nn.Embedding(
        #     image_token_offset - added_tokens_offset - self.sin_cos_numbers_offset,
        #     embed_dim,
        # )
        #######
        # self.numbers_embedding = nn.Embedding(
        #     self.sin_cos_numbers_offset, embed_dim
        # )  # TODO: remove hardcoding
        # sin_cos_values = get_sin_cos_positional_embedding(
        #     self.sin_cos_numbers_offset, embed_dim
        # )
        # self.numbers_embedding.weight.data = torch.tensor(
        # sin_cos_values, dtype=torch.bfloat16
        # )
        # self.numbers_embedding.weight.requires_grad = False
        # self.static_sin_cos_embedding = nn.Embedding(
        #     self.sin_cos_numbers_offset, embed_dim
        # )
        # self.static_sin_cos_embedding.weight.data = torch.tensor(
        #     sin_cos_values, dtype=torch.bfloat16
        # )
        # self.static_sin_cos_embedding.weight.requires_grad = False
        #######
        self.use_vqgan_codebook = use_vqgan_codebook
        self.vqgan_codebook = nn.Embedding(vqgan_vocab_size, vqgan_embed_dim)
        if self.use_vqgan_codebook:
            if self.use_mlp_projector:
                # two layer MLP projector with ReLU activation
                self.vqgan_embed_proj = nn.Sequential(
                    nn.Linear(vqgan_embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, embed_dim),
                )
            else:
                self.vqgan_embed_proj = nn.Linear(
                    vqgan_embed_dim, embed_dim, bias=False
                )
        else:
            self.vqgan_embed_proj = nn.Embedding(vqgan_vocab_size, embed_dim)
        # load the quantized weights from the path which points to a npy file
        self.vqgan_codebook.weight.data = torch.tensor(
            np.load(vqgan_codebook_path), dtype=torch.bfloat16
        )
        # make requires_grad False
        self.vqgan_codebook.weight.requires_grad = False
        self.image_token_offset = image_token_offset

    def forward(self, x):
        # x is a tensor of shape (batch_size, seq_len) containing token ids
        # separate the token ids into text and image token ids
        # text token ids are values in the range 0 to vqgan_start_index
        # image token ids are values in the range vqgan_start_index to vqgan_end_index
        # make sure to consider the shape of x and the batch size
        text_mask = (x >= 0) & (x < self.added_tokens_offset)
        # number_mask = (x >= self.added_tokens_offset) & (
        #     x < self.added_tokens_offset + self.sin_cos_numbers_offset
        # )
        # three_d_mask = (x >= self.added_tokens_offset + self.sin_cos_numbers_offset) & (
        #     x < self.vqgan_start_index
        # )
        image_mask = (x >= self.vqgan_start_index) & (x < self.vqgan_end_index)

        # assert the sum of masks is equal to the total number of tokens
        assert (
            text_mask.sum() + image_mask.sum() == x.numel()
        ), "Sum of masks is not equal to the total number of tokens"

        embeddings = torch.zeros(
            x.shape[0],
            x.shape[1],
            self.embed_dim,
            device=x.device,
            dtype=torch.bfloat16,
        )

        # Initialize tensors for text and image tokens
        text_tokens = torch.zeros_like(x)
        # numbers_tokens = torch.zeros_like(x)
        # three_d_tokens = torch.zeros_like(x)
        image_tokens = torch.zeros_like(x)

        # Fill the tensors with the corresponding token ids
        text_tokens[text_mask] = x[text_mask]
        # numbers_tokens[number_mask] = x[number_mask]
        # three_d_tokens[three_d_mask] = x[three_d_mask]
        image_tokens[image_mask] = x[image_mask]

        image_tokens -= self.image_token_offset
        # make negative values 0
        image_tokens[image_tokens < 0] = 0

        # numbers_tokens -= self.added_tokens_offset
        # make negative values 0
        # numbers_tokens[numbers_tokens < 0] = 0

        # three_d_tokens -= self.added_tokens_offset + self.sin_cos_numbers_offset
        # make negative values 0
        # three_d_tokens[three_d_tokens < 0] = 0

        # get the embeddings for the text and image tokens considering the text and image masks
        text_embeddings = self.token_embedding(text_tokens)

        if self.use_vqgan_codebook:
            image_embeddings = self.vqgan_codebook(image_tokens)
            image_embeddings = self.vqgan_embed_proj(image_embeddings)
        else:
            image_embeddings = self.vqgan_embed_proj(image_tokens)

        # numbers_embeddings = self.numbers_embedding(
        #     numbers_tokens
        # ) + self.static_sin_cos_embedding(numbers_tokens)
        # three_d_embeddings = self.added_embedding(three_d_tokens)

        embeddings[text_mask] = text_embeddings[text_mask]
        embeddings[image_mask] = image_embeddings[image_mask]
        # embeddings[number_mask] = numbers_embeddings[number_mask]
        # embeddings[three_d_mask] = three_d_embeddings[three_d_mask]

        return embeddings


# custom embedding class
class CLEVRThreeDEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        added_tokens_offset: int,
        vqgan_embed_dim: int,
        vqgan_start_index: int,
        vqgan_end_index: int,
        vqgan_vocab_size: int,
        vqgan_codebook_path: str,
        image_token_offset: int,
        use_vqgan_codebook: bool,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.added_tokens_offset = added_tokens_offset
        self.vqgan_embed_dim = vqgan_embed_dim
        self.vqgan_start_index = vqgan_start_index
        self.vqgan_end_index = vqgan_end_index
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.added_embedding = nn.Embedding(
            image_token_offset - added_tokens_offset, embed_dim
        )
        self.use_vqgan_codebook = use_vqgan_codebook
        self.vqgan_codebook = nn.Embedding(vqgan_vocab_size, vqgan_embed_dim)
        if self.use_vqgan_codebook:
            self.vqgan_embed_proj = nn.Linear(vqgan_embed_dim, embed_dim, bias=False)
        else:
            self.vqgan_embed_proj = nn.Embedding(vqgan_vocab_size, embed_dim)
        # load the quantized weights from the path which points to a npy file
        self.vqgan_codebook.weight.data = torch.tensor(
            np.load(vqgan_codebook_path), dtype=torch.bfloat16
        )
        # make requires_grad False
        self.vqgan_codebook.weight.requires_grad = False
        self.image_token_offset = image_token_offset

    def forward(self, x):
        # x is a tensor of shape (batch_size, seq_len) containing token ids
        # separate the token ids into text and image token ids
        # text token ids are values in the range 0 to vqgan_start_index
        # image token ids are values in the range vqgan_start_index to vqgan_end_index

        text_mask = (x >= 0) & (x < self.added_tokens_offset)
        three_d_mask = (x >= self.added_tokens_offset) & (x < self.vqgan_start_index)
        image_mask = (x >= self.vqgan_start_index) & (x < self.vqgan_end_index)

        embeddings = torch.zeros(
            x.shape[0],
            x.shape[1],
            self.embed_dim,
            device=x.device,
            dtype=torch.bfloat16,
        )

        # Initialize tensors for text and image tokens
        text_tokens = torch.zeros_like(x)
        three_d_tokens = torch.zeros_like(x)
        image_tokens = torch.zeros_like(x)

        # Fill the tensors with the corresponding token ids
        text_tokens[text_mask] = x[text_mask]
        three_d_tokens[three_d_mask] = x[three_d_mask]
        image_tokens[image_mask] = x[image_mask]

        image_tokens -= self.image_token_offset
        # make negative values 0
        image_tokens[image_tokens < 0] = 0

        three_d_tokens -= self.added_tokens_offset
        # make negative values 0
        three_d_tokens[three_d_tokens < 0] = 0

        # get the embeddings for the text and image tokens considering the text and image masks
        text_embeddings = self.token_embedding(text_tokens)
        if self.use_vqgan_codebook:
            image_embeddings = self.vqgan_codebook(image_tokens)
            image_embeddings = self.vqgan_embed_proj(image_embeddings)
        else:
            image_embeddings = self.vqgan_embed_proj(image_tokens)
        three_d_embeddings = self.added_embedding(three_d_tokens)

        embeddings[text_mask] = text_embeddings[text_mask]
        embeddings[image_mask] = image_embeddings[image_mask]
        embeddings[three_d_mask] = three_d_embeddings[three_d_mask]

        return embeddings


def llama3_mlp(dim: int, hidden_dim: int, quantize_base: bool = False) -> FeedForward:
    """
    Build the MLP layer associated with the Llama model.
    """
    gate_proj = (
        nn.Linear(dim, hidden_dim, bias=False)
        if not quantize_base
        else FrozenNF4Linear(dim, hidden_dim, bias=False)
    )
    down_proj = (
        nn.Linear(hidden_dim, dim, bias=False)
        if not quantize_base
        else FrozenNF4Linear(hidden_dim, dim, bias=False)
    )
    up_proj = (
        nn.Linear(dim, hidden_dim, bias=False)
        if not quantize_base
        else FrozenNF4Linear(dim, hidden_dim, bias=False)
    )
    return FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj)


# ------------------ LoRA Llama3.2 ------------------
def lora_llama3_2_clevr3d(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    *,
    # llama3.2 args
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    max_seq_len: int,
    intermediate_dim: Optional[int] = None,
    attn_dropout: float = 0.0,
    norm_eps: float = 1e-5,
    rope_base: int = 500_000,
    scale_factor: int = 32,
    # LoRA args
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    # Quantization args
    quantize_base: bool = False,
    # CLEVR3D args
    added_tokens_offset: int = 128256,
    vqgan_embed_dim: int = 256,
    vqgan_start_index: int = 129471,
    vqgan_end_index: int = 137663,
    vqgan_vocab_size: int = 8192,
    vqgan_codebook_path: str = "./3d-mllm-datasets-and-codebooks/vqgan-models-and-codebooks/domain-agnostic/quantize_weight_8192.npy",
    image_token_offset: int = 129471,
    use_vqgan_codebook: bool = True,
    use_sin_cos_numbers: bool = False,
    sin_cos_numbers_offset: int = 1201,
    use_sin_cos_plus_learned: bool = False,
) -> TransformerDecoder:
    """
    Return a version of Llama3.2 (an instance of :func:`~torchtune.modules.TransformerDecoder`)
    with LoRA applied based on the passed in configuration.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
            Default: False
        vocab_size (int): number of tokens in vocabulary.
        num_layers (int): number of layers in the transformer decoder.
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. User should ensure
            `num_heads` % `num_kv_heads` == 0. For standard MHA set `num_kv_heads` == `num_heads`,
            for GQA `num_kv_heads` < `num_heads`, and for MQA set `num_kv_heads` == 1.
        embed_dim (int): embedding dimension for self-attention
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        intermediate_dim (Optional[int]): intermediate dimension for MLP. If not specified,
            this is computed using :func:`~torchtune.modules.scale_hidden_dim_for_mlp`
        norm_eps (float): epsilon in RMS norms.
        rope_base (int): base for the rotary positional embeddings. Default: 500_000
        scale_factor (int): scaling factor for RoPE. Default: 32
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): LoRA dropout probability. Default: 0.0
        quantize_base: (bool): Whether to quantize base model weights or not. Only applied to base
            weights within linear layers LoRA is applied to. The final output linear projection is not
            supported for quantization currently.

    Returns:
        TransformerDecoder: Instantiation of Llama3.2 model with LoRA applied to
        a subset of the attention projections in each layer.

    """

    hidden_dim = (
        intermediate_dim if intermediate_dim else scale_hidden_dim_for_mlp(embed_dim)
    )
    head_dim = embed_dim // num_heads
    rope = Llama3ScaledRoPE(
        dim=head_dim, max_seq_len=max_seq_len, base=rope_base, scale_factor=scale_factor
    )
    layers = []
    for _ in range(num_layers):
        self_attn = lora_llama3_2_self_attention(
            lora_modules=lora_attn_modules,
            pos_embeddings=rope,
            head_dim=head_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_dora=use_dora,
            quantize_base=quantize_base,
        )

        if apply_lora_to_mlp:
            mlp = lora_llama3_mlp(
                dim=embed_dim,
                hidden_dim=hidden_dim,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                quantize_base=quantize_base,
                lora_dropout=lora_dropout,
                use_dora=use_dora,
            )
        else:
            mlp = llama3_mlp(
                dim=embed_dim, hidden_dim=hidden_dim, quantize_base=quantize_base
            )

        layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
            mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        )
        layers.append(layer)
    layers = nn.ModuleList(layers)
    # tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    assert not (
        use_sin_cos_numbers and use_sin_cos_plus_learned
    ), "Both use_sin_cos_numbers and use_sin_cos_plus_learned cannot be True"
    if use_sin_cos_numbers:
        tok_embeddings = CLEVRThreeDEmbeddingWithSinCosNumbers(
            vocab_size,
            embed_dim,
            added_tokens_offset,
            vqgan_embed_dim,
            vqgan_start_index,
            vqgan_end_index,
            vqgan_vocab_size,
            vqgan_codebook_path,
            image_token_offset,
            use_vqgan_codebook,
            sin_cos_numbers_offset,
        )
    elif use_sin_cos_plus_learned:
        tok_embeddings = CLEVRThreeDEmbeddingWithSinCosNumbersPlusLearned(
            vocab_size,
            embed_dim,
            added_tokens_offset,
            vqgan_embed_dim,
            vqgan_start_index,
            vqgan_end_index,
            vqgan_vocab_size,
            vqgan_codebook_path,
            image_token_offset,
            use_vqgan_codebook,
            sin_cos_numbers_offset,
        )
    else:
        tok_embeddings = CLEVRThreeDEmbedding(
            vocab_size,
            embed_dim,
            added_tokens_offset,
            vqgan_embed_dim,
            vqgan_start_index,
            vqgan_end_index,
            vqgan_vocab_size,
            vqgan_codebook_path,
            image_token_offset,
            use_vqgan_codebook,
        )

    output_proj = nn.Linear(
        embed_dim,
        vocab_size + (image_token_offset - added_tokens_offset) + vqgan_vocab_size,
        bias=False,
    )

    if apply_lora_to_output:
        raise ValueError(
            "apply_lora_to_output is currently not supporting in llama3.2 1b and 3b,"
            "as the projection layer weights are tied to the embeddings"
        )
    # output_proj = TiedLinear(tok_embeddings)

    model = TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layers=layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=(embed_dim // num_heads),
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
    )

    if quantize_base:
        # For QLoRA, we reparametrize 4-bit tensors to bf16, and offload to CPU on the fly
        # so as to not increase peak memory
        model._register_state_dict_hook(
            partial(reparametrize_as_dtype_state_dict_post_hook, offload_to_cpu=True)
        )

    return model


def lora_llama3_2_self_attention(
    lora_modules: List[LORA_ATTN_MODULES],
    pos_embeddings: nn.Module,
    *,
    # MultiHeadAttention args
    head_dim: int,
    embed_dim: int,
    num_heads: int,
    num_kv_heads: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    # LoRA args
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> MultiHeadAttention:
    """
    Return an instance of :func:`~torchtune.modules.MultiHeadAttention` with LoRA
    applied to a subset of its linear layers

    Args:
        lora_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to. Options are ``{"q_proj", "k_proj", "v_proj",
            "output_proj"}``.
        pos_embeddings (nn.Module): positional embeddings module to be passed to
            MultiHeadAttention.
        head_dim (int): dimension of each head in the multihead attention. Usually
            computed as ``embed_dim // num_heads``.
        embed_dim (int): embedding dimension for self-attention
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. User should ensure
            `num_heads` % `num_kv_heads` == 0. For standard MHA set `num_kv_heads` == `num_heads`,
            for GQA `num_kv_heads` < `num_heads`, and for MQA set `num_kv_heads` == 1.
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~torchtune.modules.KVCache`
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): LoRA dropout probability. Default: 0.0
        quantize_base (bool): Whether to quantize base model parameters for linear layers
            LoRA is being applied to. Default is ``False``.

    Returns:
        MultiHeadAttention: instantiation of self-attention module with LoRA
        applied to a subset of Q, K, V, output projections.

    Raises:
        ValueError: If lora_modules arg is an empty list
    """
    if not lora_modules:
        raise ValueError(
            f"Must pass one or more of {LORA_ATTN_MODULES} as lora_modules"
        )

    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    adapter_cls = DoRALinear if use_dora else LoRALinear
    q_proj = (
        adapter_cls(
            embed_dim,
            num_heads * head_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            quantize_base=quantize_base,
        )
        if "q_proj" in lora_modules
        else (
            nn.Linear(embed_dim, num_heads * head_dim, bias=False)
            if not quantize_base
            else FrozenNF4Linear(embed_dim, num_heads * head_dim, bias=False)
        )
    )
    k_proj = (
        adapter_cls(
            embed_dim,
            num_kv_heads * head_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            quantize_base=quantize_base,
        )
        if "k_proj" in lora_modules
        else (
            nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
            if not quantize_base
            else FrozenNF4Linear(embed_dim, num_kv_heads * head_dim, bias=False)
        )
    )
    v_proj = (
        adapter_cls(
            embed_dim,
            num_kv_heads * head_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            quantize_base=quantize_base,
        )
        if "v_proj" in lora_modules
        else (
            nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
            if not quantize_base
            else FrozenNF4Linear(embed_dim, num_kv_heads * head_dim, bias=False)
        )
    )
    output_proj = (
        adapter_cls(
            embed_dim,
            embed_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            quantize_base=quantize_base,
        )
        if "output_proj" in lora_modules
        else (
            nn.Linear(embed_dim, embed_dim, bias=False)
            if not quantize_base
            else FrozenNF4Linear(embed_dim, embed_dim, bias=False)
        )
    )

    self_attn = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_proj=q_proj,
        k_proj=k_proj,
        v_proj=v_proj,
        output_proj=output_proj,
        pos_embeddings=pos_embeddings,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
    )
    return self_attn


def lora_llama3_mlp(
    *,
    dim: int,
    hidden_dim: int,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> FeedForward:
    adapter_cls = DoRALinear if use_dora else LoRALinear
    gate_proj = adapter_cls(
        in_dim=dim,
        out_dim=hidden_dim,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        quantize_base=quantize_base,
    )
    down_proj = adapter_cls(
        in_dim=hidden_dim,
        out_dim=dim,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        quantize_base=quantize_base,
    )
    up_proj = adapter_cls(
        in_dim=dim,
        out_dim=hidden_dim,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
        quantize_base=quantize_base,
    )
    return FeedForward(
        gate_proj=gate_proj,
        down_proj=down_proj,
        up_proj=up_proj,
    )
