# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List
from functools import partial

from models.model_component_builders import (
    llama3_2_clevr3d,
    lora_llama3_2_clevr3d,
)

from torchtune.modules import TransformerDecoder
from torchtune.modules.peft import LORA_ATTN_MODULES

"""
Model builders build specific instantiations using component builders. For example
the llama3_2_1b model builder uses the llama3_2 component builder to create the
Llama3.2 1B model.
"""

DOMAIN_AGNOSTIC_VQGAN_CODEBOOK_PATH = "./kyvo-datasets-and-codebooks/vqgan-models-and-codebooks/domain-agnostic/quantize_weight_8192.npy"
CLEVR_VQGAN_CODEBOOK_PATH = "./kyvo-datasets-and-codebooks/vqgan-models-and-codebooks/clevr/custom_vqgan_embedding_1024CLEVRLARGE_256dim.npy"
OBJAWORLD_VQGAN_CODEBOOK_PATH = "./kyvo-datasets-and-codebooks/vqgan-models-and-codebooks/objaworld/custom_vqgan_embedding_256SYNTHETIC_LIVINGROOM_PARK_LARGE_EP100_256dim.npy"
OBJECTRON_VQGAN_CODEBOOK_PATH = "./kyvo-datasets-and-codebooks/vqgan-models-and-codebooks/objectron/custom_vqgan_embedding_256Omni3D-OBJECTRON_256dim.npy"


def lora_llama3_2_1b_clevr3d_sin_cos_numbers(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Llama3.2 1B model with LoRA enabled.
    The Llama3.2 defaults are the same as in :func:`~torchtune.models.llama3_2.llama3_2_1b`,
    while LoRA default params are based on
    https://github.com/tloen/alpaca-lora/blob/8bb8579e403dc78e37fe81ffbb253c413007323f/finetune.py#L41-L43.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
            Default: False
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): dropout probability for the low-rank approximation
        use_dora (bool): Decompose the LoRA weight into magnitude and direction, as
            introduced in "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of Llama3.2 1B model with LoRA applied
    """
    return lora_llama3_2_clevr3d(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=131072,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
        added_tokens_offset=128256,
        vqgan_embed_dim=256,
        vqgan_start_index=129471,
        vqgan_end_index=130495,
        vqgan_vocab_size=1024,
        vqgan_codebook_path=CLEVR_VQGAN_CODEBOOK_PATH,
        image_token_offset=129471,
        use_sin_cos_numbers=True,
    )

def llama3_2_1b_clevr3d_sin_cos_plus_learned_numbers_NO_FINDINGS() -> (
    TransformerDecoder
):
    """
    Builder for creating a Llama3.2 model initialized w/ the default 1b parameter values.

    Returns:
        TransformerDecoder: Instantiation of Llama3.2 1B model
    """
    return llama3_2_clevr3d(
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=131072,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
        added_tokens_offset=128256,
        vqgan_embed_dim=256,
        vqgan_start_index=128256,
        vqgan_end_index=136448,
        vqgan_vocab_size=8192,
        vqgan_codebook_path=DOMAIN_AGNOSTIC_VQGAN_CODEBOOK_PATH,
        image_token_offset=128256,
        use_sin_cos_numbers=False,
        sin_cos_numbers_offset=0,
        use_sin_cos_plus_learned=False,
        no_independent_numbers=False,
        no_independent_numbers_no_3d_tokens=True,
    )



def lora_llama3_2_1b_clevr3d_sin_cos_plus_learned_numbers(
    lora_attn_modules: List[LORA_ATTN_MODULES],
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
) -> TransformerDecoder:
    """
    Builder for creating a Llama3.2 1B model with LoRA enabled.
    The Llama3.2 defaults are the same as in :func:`~torchtune.models.llama3_2.llama3_2_1b`,
    while LoRA default params are based on
    https://github.com/tloen/alpaca-lora/blob/8bb8579e403dc78e37fe81ffbb253c413007323f/finetune.py#L41-L43.

    Args:
        lora_attn_modules (List[LORA_ATTN_MODULES]): list of which linear layers
            LoRA should be applied to in each self-attention block. Options are
            ``{"q_proj", "k_proj", "v_proj", "output_proj"}``.
        apply_lora_to_mlp (bool): whether to apply LoRA to the MLP in each transformer layer.
            Default: False
        apply_lora_to_output (bool): whether to apply LoRA to the model's final output projection.
            Default: False
        lora_rank (int): rank of each low-rank approximation
        lora_alpha (float): scaling factor for the low-rank approximation
        lora_dropout (float): dropout probability for the low-rank approximation
        use_dora (bool): Decompose the LoRA weight into magnitude and direction, as
            introduced in "DoRA: Weight-Decomposed Low-Rank Adaptation" (https://arxiv.org/abs/2402.09353).
        quantize_base (bool): Whether to quantize base model weights

    Returns:
        TransformerDecoder: Instantiation of Llama3.2 1B model with LoRA applied
    """
    return lora_llama3_2_clevr3d(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=131072,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
        added_tokens_offset=128256,
        vqgan_embed_dim=256,
        vqgan_start_index=129471,
        vqgan_end_index=130495,
        vqgan_vocab_size=1024,
        vqgan_codebook_path=CLEVR_VQGAN_CODEBOOK_PATH,
        image_token_offset=129471,
        use_sin_cos_numbers=False,
        use_sin_cos_plus_learned=True,
    )


def llama3_2_1b_clevr3d_sin_cos_numbers() -> TransformerDecoder:
    """
    Builder for creating a Llama3.2 model initialized w/ the default 1b parameter values.

    Returns:
        TransformerDecoder: Instantiation of Llama3.2 1B model
    """
    return llama3_2_clevr3d(
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=131072,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
        added_tokens_offset=128256,
        vqgan_embed_dim=256,
        vqgan_start_index=129471,
        vqgan_end_index=130495,
        vqgan_vocab_size=1024,
        vqgan_codebook_path=CLEVR_VQGAN_CODEBOOK_PATH,
        image_token_offset=129471,
        use_sin_cos_numbers=True,
    )


def llama3_2_1b_clevr3d_sin_cos_plus_learned_numbers_SYNTHETIC_LIVINGROOM_PARK_LARGE_EP100() -> (
    TransformerDecoder
):
    """
    Builder for creating a Llama3.2 model initialized w/ the default 1b parameter values.

    Returns:
        TransformerDecoder: Instantiation of Llama3.2 1B model
    """
    return llama3_2_clevr3d(
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=131072,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
        added_tokens_offset=128256,
        vqgan_embed_dim=256,
        vqgan_start_index=129471,
        vqgan_end_index=130495,
        vqgan_vocab_size=1024,
        vqgan_codebook_path=OBJAWORLD_VQGAN_CODEBOOK_PATH,
        image_token_offset=129471,
        use_sin_cos_numbers=False,
        use_sin_cos_plus_learned=True,
    )


def llama3_2_1b_clevr3d_sin_cos_plus_learned_numbers() -> TransformerDecoder:
    """
    Builder for creating a Llama3.2 model initialized w/ the default 1b parameter values.

    Returns:
        TransformerDecoder: Instantiation of Llama3.2 1B model
    """
    return llama3_2_clevr3d(
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=131072,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
        added_tokens_offset=128256,
        vqgan_embed_dim=256,
        vqgan_start_index=129471,
        vqgan_end_index=130495,
        vqgan_vocab_size=1024,
        vqgan_codebook_path=CLEVR_VQGAN_CODEBOOK_PATH,
        image_token_offset=129471,
        use_sin_cos_numbers=False,
        use_sin_cos_plus_learned=True,
    )


def llama3_2_1b_clevr3d_sin_cos_plus_learned_numbers_domain_agnostic_vqgan() -> (
    TransformerDecoder
):
    """
    Builder for creating a Llama3.2 model initialized w/ the default 1b parameter values.

    Returns:
        TransformerDecoder: Instantiation of Llama3.2 1B model
    """
    return llama3_2_clevr3d(
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=131072,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
        added_tokens_offset=128256,
        vqgan_embed_dim=256,
        vqgan_start_index=129471,
        vqgan_end_index=137663,
        vqgan_vocab_size=8192,
        vqgan_codebook_path=DOMAIN_AGNOSTIC_VQGAN_CODEBOOK_PATH,
        image_token_offset=129471,
        use_sin_cos_numbers=False,
        use_sin_cos_plus_learned=True,
    )


def llama3_2_1b_clevr3d_sin_cos_plus_learned_numbers_mlp_projector() -> (
    TransformerDecoder
):
    """
    Builder for creating a Llama3.2 model initialized w/ the default 1b parameter values.

    Returns:
        TransformerDecoder: Instantiation of Llama3.2 1B model
    """
    return llama3_2_clevr3d(
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=131072,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
        added_tokens_offset=128256,
        vqgan_embed_dim=256,
        vqgan_start_index=129471,
        vqgan_end_index=130495,
        vqgan_vocab_size=1024,
        vqgan_codebook_path=CLEVR_VQGAN_CODEBOOK_PATH,
        image_token_offset=129471,
        use_sin_cos_numbers=False,
        use_sin_cos_plus_learned=True,
        use_mlp_projector=True,
    )


def llama3_2_3b_clevr3d_sin_cos_plus_learned_numbers() -> TransformerDecoder:
    """
    Builder for creating a Llama3.2 model initialized w/ the default 1b parameter values.

    Returns:
        TransformerDecoder: Instantiation of Llama3.2 1B model
    """
    return llama3_2_clevr3d(
        vocab_size=128_256,
        num_layers=28,
        num_heads=24,
        num_kv_heads=8,
        embed_dim=3072,
        max_seq_len=131072,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
        added_tokens_offset=128256,
        vqgan_embed_dim=256,
        vqgan_start_index=129471,
        vqgan_end_index=130495,
        vqgan_vocab_size=1024,
        vqgan_codebook_path=CLEVR_VQGAN_CODEBOOK_PATH,
        image_token_offset=129471,
        use_sin_cos_numbers=False,
        use_sin_cos_plus_learned=True,
    )


def llama3_2_1b_clevr3d_NO_sin_cos_numbers() -> TransformerDecoder:
    """
    Builder for creating a Llama3.2 model initialized w/ the default 1b parameter values.

    Returns:
        TransformerDecoder: Instantiation of Llama3.2 1B model
    """
    return llama3_2_clevr3d(
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=131072,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
        added_tokens_offset=128256,
        vqgan_embed_dim=256,
        vqgan_start_index=129471,
        vqgan_end_index=130495,
        vqgan_vocab_size=1024,
        vqgan_codebook_path=CLEVR_VQGAN_CODEBOOK_PATH,
        image_token_offset=129471,
        use_sin_cos_numbers=False,
    )


def llama3_2_1b_clevr3d_sin_cos_plus_learned_numbers_omni3d_objectron_custom_finer() -> (
    TransformerDecoder
):
    """
    Builder for creating a Llama3.2 model initialized w/ the default 1b parameter values.

    Returns:
        TransformerDecoder: Instantiation of Llama3.2 1B model
    """
    return llama3_2_clevr3d(
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=131072,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
        added_tokens_offset=128256,
        vqgan_embed_dim=256,
        vqgan_start_index=128372,
        vqgan_end_index=129396,
        vqgan_vocab_size=1024,
        vqgan_codebook_path=OBJECTRON_VQGAN_CODEBOOK_PATH,
        image_token_offset=128372,
        use_sin_cos_numbers=False,
        sin_cos_numbers_offset=104,
        use_sin_cos_plus_learned=True,
    )
