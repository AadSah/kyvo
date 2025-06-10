# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re

from typing import Any, Dict

import torch

_FROM_META_3D_SIN_COS_NUM = {
    "tok_embeddings.token_embedding.weight": "tok_embeddings.token_embedding.weight",
    "tok_embeddings.numbers_embedding.weight": "tok_embeddings.numbers_embedding.weight",
    "tok_embeddings.added_embedding.weight": "tok_embeddings.added_embedding.weight",
    "tok_embeddings.vqgan_codebook.weight": "tok_embeddings.vqgan_codebook.weight",
    "tok_embeddings.vqgan_embed_proj.weight": "tok_embeddings.vqgan_embed_proj.weight",
    "norm.weight": "norm.scale",
    "output.weight": "output.weight",
    "layers.{}.attention.wk.weight": "layers.{}.attn.k_proj.weight",
    "layers.{}.attention.wq.weight": "layers.{}.attn.q_proj.weight",
    "layers.{}.attention.wv.weight": "layers.{}.attn.v_proj.weight",
    "layers.{}.attention.wo.weight": "layers.{}.attn.output_proj.weight",
    "layers.{}.attention_norm.weight": "layers.{}.sa_norm.scale",
    "layers.{}.ffn_norm.weight": "layers.{}.mlp_norm.scale",
    "layers.{}.feed_forward.w1.weight": "layers.{}.mlp.w1.weight",
    "layers.{}.feed_forward.w2.weight": "layers.{}.mlp.w2.weight",
    "layers.{}.feed_forward.w3.weight": "layers.{}.mlp.w3.weight",
}

_FROM_META_3D_SIN_COS_PLUS_LEARNED_NUM = {
    "tok_embeddings.token_embedding.weight": "tok_embeddings.token_embedding.weight",
    "tok_embeddings.numbers_embedding.weight": "tok_embeddings.numbers_embedding.weight",
    "tok_embeddings.static_sin_cos_embedding.weight": "tok_embeddings.static_sin_cos_embedding.weight",
    "tok_embeddings.added_embedding.weight": "tok_embeddings.added_embedding.weight",
    "tok_embeddings.vqgan_codebook.weight": "tok_embeddings.vqgan_codebook.weight",
    "tok_embeddings.vqgan_embed_proj.weight": "tok_embeddings.vqgan_embed_proj.weight",
    "norm.weight": "norm.scale",
    "output.weight": "output.weight",
    "layers.{}.attention.wk.weight": "layers.{}.attn.k_proj.weight",
    "layers.{}.attention.wq.weight": "layers.{}.attn.q_proj.weight",
    "layers.{}.attention.wv.weight": "layers.{}.attn.v_proj.weight",
    "layers.{}.attention.wo.weight": "layers.{}.attn.output_proj.weight",
    "layers.{}.attention_norm.weight": "layers.{}.sa_norm.scale",
    "layers.{}.ffn_norm.weight": "layers.{}.mlp_norm.scale",
    "layers.{}.feed_forward.w1.weight": "layers.{}.mlp.w1.weight",
    "layers.{}.feed_forward.w2.weight": "layers.{}.mlp.w2.weight",
    "layers.{}.feed_forward.w3.weight": "layers.{}.mlp.w3.weight",
}

_FROM_META_3D_SIN_COS_PLUS_LEARNED_NUM_WITH_MLP = {
    "tok_embeddings.token_embedding.weight": "tok_embeddings.token_embedding.weight",
    "tok_embeddings.numbers_embedding.weight": "tok_embeddings.numbers_embedding.weight",
    "tok_embeddings.static_sin_cos_embedding.weight": "tok_embeddings.static_sin_cos_embedding.weight",
    "tok_embeddings.added_embedding.weight": "tok_embeddings.added_embedding.weight",
    "tok_embeddings.vqgan_codebook.weight": "tok_embeddings.vqgan_codebook.weight",
    "tok_embeddings.vqgan_embed_proj.{}.weight": "tok_embeddings.vqgan_embed_proj.{}.weight",
    "tok_embeddings.vqgan_embed_proj.{}.bias": "tok_embeddings.vqgan_embed_proj.{}.bias",
    "tok_embeddings.vqgan_embed_proj.{}.weight": "tok_embeddings.vqgan_embed_proj.{}.weight",
    "tok_embeddings.vqgan_embed_proj.{}.bias": "tok_embeddings.vqgan_embed_proj.{}.bias",
    "norm.weight": "norm.scale",
    "output.weight": "output.weight",
    "layers.{}.attention.wk.weight": "layers.{}.attn.k_proj.weight",
    "layers.{}.attention.wq.weight": "layers.{}.attn.q_proj.weight",
    "layers.{}.attention.wv.weight": "layers.{}.attn.v_proj.weight",
    "layers.{}.attention.wo.weight": "layers.{}.attn.output_proj.weight",
    "layers.{}.attention_norm.weight": "layers.{}.sa_norm.scale",
    "layers.{}.ffn_norm.weight": "layers.{}.mlp_norm.scale",
    "layers.{}.feed_forward.w1.weight": "layers.{}.mlp.w1.weight",
    "layers.{}.feed_forward.w2.weight": "layers.{}.mlp.w2.weight",
    "layers.{}.feed_forward.w3.weight": "layers.{}.mlp.w3.weight",
}

_FROM_META_3D_SIN_COS_NUM_WITH_MLP = {
    "tok_embeddings.token_embedding.weight": "tok_embeddings.token_embedding.weight",
    "tok_embeddings.numbers_embedding.weight": "tok_embeddings.numbers_embedding.weight",
    "tok_embeddings.added_embedding.weight": "tok_embeddings.added_embedding.weight",
    "tok_embeddings.vqgan_codebook.weight": "tok_embeddings.vqgan_codebook.weight",
    "tok_embeddings.vqgan_embed_proj.{}.weight": "tok_embeddings.vqgan_embed_proj.{}.weight",
    "tok_embeddings.vqgan_embed_proj.{}.bias": "tok_embeddings.vqgan_embed_proj.{}.bias",
    "tok_embeddings.vqgan_embed_proj.{}.weight": "tok_embeddings.vqgan_embed_proj.{}.weight",
    "tok_embeddings.vqgan_embed_proj.{}.bias": "tok_embeddings.vqgan_embed_proj.{}.bias",
    "norm.weight": "norm.scale",
    "output.weight": "output.weight",
    "layers.{}.attention.wk.weight": "layers.{}.attn.k_proj.weight",
    "layers.{}.attention.wq.weight": "layers.{}.attn.q_proj.weight",
    "layers.{}.attention.wv.weight": "layers.{}.attn.v_proj.weight",
    "layers.{}.attention.wo.weight": "layers.{}.attn.output_proj.weight",
    "layers.{}.attention_norm.weight": "layers.{}.sa_norm.scale",
    "layers.{}.ffn_norm.weight": "layers.{}.mlp_norm.scale",
    "layers.{}.feed_forward.w1.weight": "layers.{}.mlp.w1.weight",
    "layers.{}.feed_forward.w2.weight": "layers.{}.mlp.w2.weight",
    "layers.{}.feed_forward.w3.weight": "layers.{}.mlp.w3.weight",
}

_FROM_META_3D = {
    "tok_embeddings.token_embedding.weight": "tok_embeddings.token_embedding.weight",
    "tok_embeddings.added_embedding.weight": "tok_embeddings.added_embedding.weight",
    "tok_embeddings.vqgan_codebook.weight": "tok_embeddings.vqgan_codebook.weight",
    "tok_embeddings.vqgan_embed_proj.weight": "tok_embeddings.vqgan_embed_proj.weight",
    "norm.weight": "norm.scale",
    "output.weight": "output.weight",
    "layers.{}.attention.wk.weight": "layers.{}.attn.k_proj.weight",
    "layers.{}.attention.wq.weight": "layers.{}.attn.q_proj.weight",
    "layers.{}.attention.wv.weight": "layers.{}.attn.v_proj.weight",
    "layers.{}.attention.wo.weight": "layers.{}.attn.output_proj.weight",
    "layers.{}.attention_norm.weight": "layers.{}.sa_norm.scale",
    "layers.{}.ffn_norm.weight": "layers.{}.mlp_norm.scale",
    "layers.{}.feed_forward.w1.weight": "layers.{}.mlp.w1.weight",
    "layers.{}.feed_forward.w2.weight": "layers.{}.mlp.w2.weight",
    "layers.{}.feed_forward.w3.weight": "layers.{}.mlp.w3.weight",
}

# state dict key mappings from Meta's format to torchtune's format
_FROM_META = {
    "tok_embeddings.weight": "tok_embeddings.weight",
    "norm.weight": "norm.scale",
    "output.weight": "output.weight",
    "layers.{}.attention.wk.weight": "layers.{}.attn.k_proj.weight",
    "layers.{}.attention.wq.weight": "layers.{}.attn.q_proj.weight",
    "layers.{}.attention.wv.weight": "layers.{}.attn.v_proj.weight",
    "layers.{}.attention.wo.weight": "layers.{}.attn.output_proj.weight",
    "layers.{}.attention_norm.weight": "layers.{}.sa_norm.scale",
    "layers.{}.ffn_norm.weight": "layers.{}.mlp_norm.scale",
    "layers.{}.feed_forward.w1.weight": "layers.{}.mlp.w1.weight",
    "layers.{}.feed_forward.w2.weight": "layers.{}.mlp.w2.weight",
    "layers.{}.feed_forward.w3.weight": "layers.{}.mlp.w3.weight",
}

# state dict key mappings from HF's format to torchtune's format
_FROM_HF = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attn.q_proj.weight",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attn.k_proj.weight",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attn.v_proj.weight",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attn.output_proj.weight",
    "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.mlp.w1.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.mlp.w3.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.mlp.w2.weight",
    "model.layers.{}.input_layernorm.weight": "layers.{}.sa_norm.scale",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.mlp_norm.scale",
    "model.norm.weight": "norm.scale",
    "lm_head.weight": "output.weight",
}


def get_mapped_key(key: str, mapping_dict: Dict[str, str]) -> str:
    try:
        # Checks if there is a layer # in the key
        if any(k.isdigit() for k in key.split(".")):
            # Replace layer number with "{}" to create key for lookup
            abstract_key = re.sub(r"(\.\d+)", ".{}", key)
            layer_num = re.search(r"\d+", key).group(0)
            new_key = mapping_dict[abstract_key]
            new_key = new_key.format(layer_num)
        else:
            new_key = mapping_dict[key]
    except KeyError as e:
        raise Exception(
            f'Error converting the state dict. Found unexpected key: "{key}". '
            "Please make sure you're loading a checkpoint with the right format. "
        ) from e

    return new_key


def meta_to_tune(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from Meta's format to torchtune's format. State dicts
    from multiple checkpoint files should be consolidated into a single state dict
    before calling this function.

    Eg of Meta-format state dict can be found in the ``meta-llama/Llama-2-7b``
    repo in HF (https://huggingface.co/meta-llama/Llama-2-7b).

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in Meta's format.

    Returns:
        Dict[str, torch.Tensor]: State dict in torchtune's format.
    """
    converted_state_dict = {}
    for key, value in state_dict.items():
        if key not in ["rope.freqs"]:  # Skip loading the position embeddings
            new_key = get_mapped_key(key, _FROM_META)
            converted_state_dict[new_key] = value

    return converted_state_dict


def meta_to_tune_3d(
    state_dict: Dict[str, torch.Tensor], convert_weights_type: str
) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from Meta's format to torchtune's format. State dicts
    from multiple checkpoint files should be consolidated into a single state dict
    before calling this function.

    Eg of Meta-format state dict can be found in the ``meta-llama/Llama-2-7b``
    repo in HF (https://huggingface.co/meta-llama/Llama-2-7b).

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in Meta's format.

    Returns:
        Dict[str, torch.Tensor]: State dict in torchtune's format.
    """
    if convert_weights_type == "3d":
        dictionary_to_use = _FROM_META_3D
    elif convert_weights_type == "3d_sin_cos_num":
        dictionary_to_use = _FROM_META_3D_SIN_COS_NUM
    elif convert_weights_type == "3d_sin_cos_plus_learned_num":
        dictionary_to_use = _FROM_META_3D_SIN_COS_PLUS_LEARNED_NUM
    elif convert_weights_type == "3d_sin_cos_plus_learned_num_with_mlp":
        dictionary_to_use = _FROM_META_3D_SIN_COS_PLUS_LEARNED_NUM_WITH_MLP
    elif convert_weights_type == "3d_sin_cos_num_with_mlp":
        dictionary_to_use = _FROM_META_3D_SIN_COS_NUM_WITH_MLP
    else:
        raise ValueError(
            "convert_weights_type should be one of '3d', '3d_sin_cos_num', '3d_sin_cos_plus_learned_num', '3d_sin_cos_plus_learned_num_with_mlp', '3d_sin_cos_num_with_mlp'"
        )
    converted_state_dict = {}
    for key, value in state_dict.items():
        if key not in ["rope.freqs"]:  # Skip loading the position embeddings
            new_key = get_mapped_key(key, dictionary_to_use)
            converted_state_dict[new_key] = value

    return converted_state_dict


def tune_to_meta(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from torchtune's format to Meta's format. This function
    doesn't handle any sharding or splitting of state dicts. It follows the
    state_dict IN -> state_dict OUT pattern.

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in torchtune's format.

    Returns:
        Dict[str, torch.Tensor]: State dict in Meta's format.
    """
    converted_state_dict = {}
    inverted_mapping_dict = {v: k for k, v in _FROM_META.items()}

    for key, value in state_dict.items():
        new_key = get_mapped_key(key, inverted_mapping_dict)
        converted_state_dict[new_key] = value

    return converted_state_dict


def tune_to_meta_3d(
    state_dict: Dict[str, torch.Tensor], convert_weights_type: str
) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from torchtune's format to Meta's format. This function
    doesn't handle any sharding or splitting of state dicts. It follows the
    state_dict IN -> state_dict OUT pattern.

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in torchtune's format.

    Returns:
        Dict[str, torch.Tensor]: State dict in Meta's format.
    """
    if convert_weights_type == "3d":
        dictionary_to_use = _FROM_META_3D
    elif convert_weights_type == "3d_sin_cos_num":
        dictionary_to_use = _FROM_META_3D_SIN_COS_NUM
    elif convert_weights_type == "3d_sin_cos_plus_learned_num":
        dictionary_to_use = _FROM_META_3D_SIN_COS_PLUS_LEARNED_NUM
    elif convert_weights_type == "3d_sin_cos_plus_learned_num_with_mlp":
        dictionary_to_use = _FROM_META_3D_SIN_COS_PLUS_LEARNED_NUM_WITH_MLP
    elif convert_weights_type == "3d_sin_cos_num_with_mlp":
        dictionary_to_use = _FROM_META_3D_SIN_COS_NUM_WITH_MLP
    else:
        raise ValueError(
            "convert_weights_type should be one of '3d', '3d_sin_cos_num', '3d_sin_cos_plus_learned_num', '3d_sin_cos_plus_learned_num_with_mlp', '3d_sin_cos_num_with_mlp'"
        )
    converted_state_dict = {}
    inverted_mapping_dict = {v: k for k, v in dictionary_to_use.items()}

    for key, value in state_dict.items():
        new_key = get_mapped_key(key, inverted_mapping_dict)
        converted_state_dict[new_key] = value

    return converted_state_dict
