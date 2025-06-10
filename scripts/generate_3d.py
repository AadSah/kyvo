# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from typing import Any, Dict, List, Optional, Union

import torch
from omegaconf import DictConfig
from torch import nn

from torchtune import config, generation, training, utils
from torchtune.config._utils import _get_component_from_path
from torchtune.data import ChatFormat, InstructTemplate, Message

import numpy as np
from transformers import AutoTokenizer
import json
import math
import re
import os
import tokenizer
import generate_function

logger = utils.get_logger("DEBUG")


class InferenceRecipe:
    """
    Recipe for generating tokens from a dense Transformer-based LLM.

    Currently this recipe supports single-GPU generation only. Speculative
    decoding is not supported.

    For more details on how to use this recipe for generation, please see our
    tutorial: https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#generation

    For using this recipe with a quantized model, please the following section of
    the above tutorial:
    https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#speeding-up-generation-using-quantization
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(dtype=cfg.dtype, device=self._device)
        self._quantizer = config.instantiate(cfg.quantizer)
        self._quantization_mode = training.get_quantizer_mode(self._quantizer)

        self.DOMAIN_AGNOSTIC_VQGAN_CODEBOOK_PATH = "./3d-mllm-datasets-and-codebooks/vqgan-models-and-codebooks/domain-agnostic/quantize_weight_8192.npy"
        self.CLEVR_VQGAN_CODEBOOK_PATH = "./3d-mllm-datasets-and-codebooks/vqgan-models-and-codebooks/clevr/custom_vqgan_embedding_1024CLEVRLARGE_256dim.npy"
        self.OBJAWORLD_VQGAN_CODEBOOK_PATH = "./3d-mllm-datasets-and-codebooks/vqgan-models-and-codebooks/objaworld/custom_vqgan_embedding_256SYNTHETIC_LIVINGROOM_PARK_LARGE_EP100_256dim.npy"
        self.OBJECTRON_VQGAN_CODEBOOK_PATH = "./3d-mllm-datasets-and-codebooks/vqgan-models-and-codebooks/objectron/custom_vqgan_embedding_256Omni3D-OBJECTRON_256dim.npy"

        training.set_seed(seed=cfg.seed)

    def reverse_reordered_list(self, reordered):
        n = len(reordered)
        center = int((n // 2) - (math.sqrt(n) // 2))
        original = [0] * n
        original[center] = reordered[0]
        left, right = center - 1, center + 1
        index = 1
        while left >= 0 or right < n:
            if left >= 0:
                original[left] = reordered[index]
                left -= 1
                index += 1
            if right < n:
                original[right] = reordered[index]
                right += 1
                index += 1

        return original

    def reorder_list_optimized(self, data):
        n = len(data)
        center = int((n // 2) - (math.sqrt(n) // 2))
        reordered = [0] * n
        reordered[0] = data[center]
        left, right = center - 1, center + 1
        index = 1
        while left >= 0 or right < n:
            if left >= 0:
                reordered[index] = data[left]
                left -= 1
                index += 1
            if right < n:
                reordered[index] = data[right]
                right += 1
                index += 1

        return reordered

    def setup(self, cfg: DictConfig) -> None:
        checkpointer = config.instantiate(cfg.checkpointer)
        if self._quantization_mode is None:
            ckpt_dict = checkpointer.load_checkpoint_3d()
        else:
            # weights_only needs to be False when loading a quantized model
            # currently loading a quantized model is only supported with the
            # FullModelTorchTuneCheckpointer
            ckpt_dict = checkpointer.load_checkpoint_3d(weights_only=False)

        self._model = self._setup_model(
            model_cfg=cfg.model,
            model_state_dict=ckpt_dict[training.MODEL_KEY],
        )

        self.dataset_name = cfg.dataset_name
        if self.dataset_name == "Objectron":
            self._tokenizer = tokenizer.get_tokenizer_omni3d_objectron()
        else:
            self._tokenizer = tokenizer.get_tokenizer()

    def _setup_model(
        self,
        model_cfg: DictConfig,
        model_state_dict: Dict[str, Any],
    ) -> nn.Module:
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)

        if self._quantization_mode is not None:
            model = self._quantizer.quantize(model)
            model = model.to(device=self._device, dtype=self._dtype)

        model.load_state_dict(model_state_dict)

        # Validate model was loaded in with the expected dtype.
        training.validate_expected_param_dtype(
            model.named_parameters(), dtype=self._dtype
        )
        logger.info(f"Model is initialized with precision {self._dtype}.")

        return model

    def _setup_data(self, cfg: DictConfig) -> None:

        self.text_data = None
        self.image_data = None
        self.three_d_data = None
        self.text_target_data = None
        self.image_target_data = None
        self.three_d_target_data = None

        with open(cfg.text_file, "r") as f:
            text_data = json.load(f)

        self.text_tokens = {k: v["token_ids"] for k, v in sorted(text_data.items())}
        self.text_data_keys = list(sorted(self.text_tokens.keys()))

        if cfg.load_image_data:
            with open(cfg.image_file, "r") as f:
                image_data = json.load(f)
            self.image_tokens = {k: image_data[k] for k in self.text_data_keys}

        if cfg.load_image_target_data:
            with open(cfg.image_target_file, "r") as f:
                image_target_data = json.load(f)
            self.image_target_tokens = {
                k: image_target_data[k] for k in self.text_data_keys
            }

        if cfg.load_three_d_data:
            with open(cfg.three_d_file, "r") as f:
                three_d_data = json.load(f)
            self.three_d_tokens = {
                k: v["token_ids"]
                for k, v in sorted(three_d_data.items())
                if k in self.text_data_keys
            }

        if cfg.load_three_d_target_data:
            with open(cfg.three_d_target_file, "r") as f:
                three_d_target_data = json.load(f)
            self.three_d_target_tokens = {
                k: v["token_ids"]
                for k, v in sorted(three_d_target_data.items())
                if k in self.text_data_keys
            }

        if cfg.load_text_target_data:
            with open(cfg.text_target_file, "r") as f:
                text_target_data = json.load(f)
            self.text_target_tokens = {
                k: text_target_data[k]["token_ids"] for k in self.text_data_keys
            }

        if cfg.dataset_name == "Objectron":
            self.image_file_paths = {
                k: v["string"]["file_path"]
                for k, v in sorted(three_d_data.items())
                if k in self.text_data_keys
            }

    def get_objects_from_text(self, text):
        extracted_objects = []

        text = text.replace("<|end_of_text|>", "")
        text = text.replace("[SCENE-START]", "")
        text = text.replace("[SCENE-END]", "")

        object_starts = [m.start() for m in re.finditer(r"\[OBJECT-START\]", text)]
        object_ends = [m.start() for m in re.finditer(r"\[OBJECT-END\]", text)]

        for i in range(len(object_starts)):
            try:
                start = object_starts[i]
                end = object_ends[i]
                object_str = text[start : end + len("[OBJECT-END]")]

                object_info = {}

                if self.dataset_name == "CLEVR":
                    size_match = re.search(r"\[SIZE\] (\w+)", object_str)
                    color_match = re.search(r"\[COLOR\] (\w+)", object_str)
                    material_match = re.search(r"\[MATERIAL\] (\w+)", object_str)
                    shape_match = re.search(r"\[SHAPE\] (\w+)", object_str)
                    location_match = re.findall(r"-?\d+\.\d{3}", object_str)
                    object_info["size"] = size_match.group(1)
                    object_info["color"] = color_match.group(1)
                    object_info["material"] = material_match.group(1)
                    object_info["shape"] = shape_match.group(1)
                    object_info["3d_coords"] = [float(x) for x in location_match]
                elif self.dataset_name == "ObjaWorld":
                    shape_match = re.search(r"\[SHAPE\] (\w+)", object_str)
                    location_match = re.findall(r"-?\d+\.\d{3}", object_str)
                    object_info["shape"] = shape_match.group(1)
                    object_info["3d_coords"] = [float(x) for x in location_match]
                elif self.dataset_name == "Objectron":
                    category_match = re.search(r"\[CATEGORY\] (\w+)", object_str)
                    center_cam_match = re.search(
                        r"\[CENTER_CAM\](-?\d+\.\d{2})(-?\d+\.\d{2})(\d+\.\d{2})",
                        object_str,
                    )
                    dimensions_match = re.search(
                        r"\[DIMENSIONS\](-?\d+\.\d{2})(-?\d+\.\d{2})(\d+\.\d{2})",
                        object_str,
                    )
                    object_info["category"] = category_match.group(1)
                    object_info["center_cam"] = [
                        float(center_cam_match.group(1)),
                        float(center_cam_match.group(2)),
                        float(center_cam_match.group(3)),
                    ]
                    object_info["dimensions"] = [
                        float(dimensions_match.group(1)),
                        float(dimensions_match.group(2)),
                        float(dimensions_match.group(3)),
                    ]
                else:
                    raise ValueError("Dataset not supported")

                extracted_objects.append(object_info)
            except:
                continue

        return extracted_objects

    def construct_sequence(
        self,
        text_sample,
        image_sample,
        three_d_sample,
        image_target_sample,
        three_d_target_sample,
        text_target_sample,
        cfg,
    ):

        if cfg.task_type == "I+3+T-I+3":
            original_scene_tokens = three_d_sample
            tokens = (
                self.BOS
                + self.BOIMG
                + image_sample
                + self.EOIMG
                + three_d_sample
                + text_sample
                + self.OUTSEP
            )
            groundtruth_target_tokens = (
                self.BOIMG
                + image_target_sample
                + self.EOIMG
                + three_d_target_sample
                + self.EOS
            )
        elif cfg.task_type == "I+Q-A":
            original_scene_tokens = three_d_sample
            tokens = (
                self.BOS
                + self.BOIMG
                + image_sample
                + self.EOIMG
                + text_sample
                + self.OUTSEP
            )
            groundtruth_target_tokens = text_target_sample + self.EOS
        elif cfg.task_type == "I+3+Q-A":
            original_scene_tokens = three_d_sample
            tokens = (
                self.BOS
                + self.BOIMG
                + image_sample
                + self.EOIMG
                + three_d_sample
                + text_sample
                + self.OUTSEP
            )
            groundtruth_target_tokens = text_target_sample + self.EOS
        elif cfg.task_type == "3+I+Q-A":
            original_scene_tokens = three_d_sample
            tokens = (
                self.BOS
                + three_d_sample
                + self.BOIMG
                + image_sample
                + self.EOIMG
                + text_sample
                + self.OUTSEP
            )
            groundtruth_target_tokens = text_target_sample + self.EOS
        elif cfg.task_type == "3+I+T-3+I":
            original_scene_tokens = three_d_sample
            tokens = (
                self.BOS
                + three_d_sample
                + self.BOIMG
                + image_sample
                + self.EOIMG
                + text_sample
                + self.OUTSEP
            )
            groundtruth_target_tokens = (
                three_d_target_sample
                + self.BOIMG
                + image_target_sample
                + self.EOIMG
                + self.EOS
            )
        elif cfg.task_type == "3-I":
            original_scene_tokens = three_d_sample
            tokens = self.BOS + three_d_sample + self.OUTSEP
            groundtruth_target_tokens = (
                self.BOIMG + image_sample + self.EOIMG + self.EOS
            )
        elif cfg.task_type == "I-3":
            original_scene_tokens = three_d_sample
            tokens = self.BOS + self.BOIMG + image_sample + self.EOIMG + self.OUTSEP
            groundtruth_target_tokens = three_d_sample + self.EOS

        return tokens, groundtruth_target_tokens, original_scene_tokens

    @torch.inference_mode()
    def generate(self, cfg: DictConfig):

        self._setup_data(cfg=cfg)

        groundtruth_scenes = []
        predicted_scenes = []
        original_scenes = []

        groundtruth_answers = []
        predicted_answers = []

        # if there is no num_samples, then set it to the length of the text data
        if cfg.num_samples == -1:
            cfg.num_samples = len(self.text_data_keys)

        self.BOS = [128000]
        self.EOS = [128001]
        if cfg.dataset_name == "Objectron":
            self.BOIMG = [128366]
            self.EOIMG = [128367]
            self.OUTSEP = [128370]
            assert cfg.image_token_offset == 128372, "Image token offset is incorrect"
        else:
            self.BOIMG = [129466]
            self.EOIMG = [129467]
            self.OUTSEP = [129470]
            assert cfg.image_token_offset == 129471, "Image token offset is incorrect"

        # iterate over the samples
        for sample_id in range(
            cfg.sample_start_idx, cfg.sample_start_idx + cfg.num_samples
        ):
            print(
                f"Processing sample: {sample_id+1}/{cfg.num_samples} with key: {self.text_data_keys[sample_id]}"
            )

            text_sample = image_sample = three_d_sample = None
            image_target_sample = three_d_target_sample = text_target_sample = None

            text_sample = self.text_tokens[self.text_data_keys[sample_id]]

            if cfg.load_image_data:
                image_sample = self.image_tokens[self.text_data_keys[sample_id]]
                image_sample = [x + cfg.image_token_offset for x in image_sample]
                if (
                    cfg.reorder_image_tokens
                    and cfg.task_type != "I+3+T-I+3"
                    and cfg.task_type != "3+I+T-3+I"
                ):
                    print("Reordering image tokens (source)")
                    image_sample = self.reorder_list_optimized(image_sample)

            if cfg.load_image_target_data:
                image_target_sample = self.image_target_tokens[
                    self.text_data_keys[sample_id]
                ]
                image_target_sample = [
                    x + cfg.image_token_offset for x in image_target_sample
                ]
                if cfg.reorder_image_tokens:
                    print("Reordering image tokens (target)")
                    image_target_sample = self.reorder_list_optimized(
                        image_target_sample
                    )

            if cfg.load_three_d_data:
                three_d_sample = self.three_d_tokens[self.text_data_keys[sample_id]]

            if cfg.load_three_d_target_data:
                three_d_target_sample = self.three_d_target_tokens[
                    self.text_data_keys[sample_id]
                ]

            if cfg.load_text_target_data:
                text_target_sample = self.text_target_tokens[
                    self.text_data_keys[sample_id]
                ]

            tokens, groundtruth_target_tokens, original_scene_tokens = (
                self.construct_sequence(
                    text_sample,
                    image_sample,
                    three_d_sample,
                    image_target_sample,
                    three_d_target_sample,
                    text_target_sample,
                    cfg,
                )
            )

            prompt = torch.tensor(tokens, dtype=torch.int, device=self._device)
            custom_generate_next_token = None

            t0 = time.perf_counter()
            generated_tokens, generated_tokens_logits = generate_function.generate(
                model=self._model,
                prompt=prompt,
                max_generated_tokens=cfg.max_new_tokens,
                pad_id=0,
                temperature=cfg.temperature,
                stop_tokens=self.EOS,
                custom_generate_next_token=custom_generate_next_token,
            )
            t = time.perf_counter() - t0

            only_generated_tokens = []
            only_generated_image_tokens = []
            for token in generated_tokens[0][prompt.size(0) :]:
                only_generated_tokens.append(token)
                if token >= cfg.image_token_offset:
                    only_generated_image_tokens.append(token)
            only_generated_tokens = torch.tensor(
                only_generated_tokens, dtype=torch.int, device=self._device
            ).unsqueeze(0)

            if cfg.load_text_target_data:
                groundtruth_answers.append(
                    {
                        "image_filename": f"groundtruth_scene_{sample_id}",
                        "answer": self._tokenizer.decode(groundtruth_target_tokens),
                        "key": self.text_data_keys[sample_id],
                    }
                )
                predicted_answers.append(
                    {
                        "image_filename": f"predicted_scene_{sample_id}",
                        "answer": self._tokenizer.decode(
                            only_generated_tokens.tolist()[0]
                        ),
                        "key": self.text_data_keys[sample_id],
                    }
                )

            groundtruth_objects = self.get_objects_from_text(
                self._tokenizer.decode(groundtruth_target_tokens)
            )

            generated_tokens_decoded = self._tokenizer.decode(generated_tokens[0])
            generated_objects = self.get_objects_from_text(
                generated_tokens_decoded.split("[OUTPUT-START]")[1]
            )

            original_scene_objects = None
            if original_scene_tokens is not None:
                original_scene_tokens = torch.tensor(
                    original_scene_tokens, dtype=torch.int, device=self._device
                )
                original_scene_objects = self.get_objects_from_text(
                    self._tokenizer.decode(original_scene_tokens)
                )

            groundtruth_scenes.append(
                {
                    "image_filename": f"groundtruth_scene_{sample_id}",
                    "objects": groundtruth_objects,
                    "key": (
                        self.text_data_keys[sample_id]
                        if cfg.dataset_name != "Objectron"
                        else self.image_file_paths[self.text_data_keys[sample_id]]
                    ),
                }
            )
            predicted_scenes.append(
                {
                    "image_filename": f"predicted_scene_{sample_id}",
                    "objects": generated_objects,
                    "key": (
                        self.text_data_keys[sample_id]
                        if cfg.dataset_name != "Objectron"
                        else self.image_file_paths[self.text_data_keys[sample_id]]
                    ),
                }
            )
            original_scenes.append(
                {
                    "image_filename": f"original_scene_{sample_id}",
                    "objects": original_scene_objects,
                    "key": (
                        self.text_data_keys[sample_id]
                        if cfg.dataset_name != "Objectron"
                        else self.image_file_paths[self.text_data_keys[sample_id]]
                    ),
                }
            )

            # for image tokens
            if cfg.vqgan_type == "domain-agnostic":
                vqgan_codebook = nn.Embedding(8192, 256)
                vqgan_codebook.weight.data = torch.tensor(
                    np.load(self.DOMAIN_AGNOSTIC_VQGAN_CODEBOOK_PATH)
                )
            elif cfg.vqgan_type == "clevr":
                vqgan_codebook = nn.Embedding(1024, 256)
                vqgan_codebook.weight.data = torch.tensor(
                    np.load(self.CLEVR_VQGAN_CODEBOOK_PATH)
                )
            elif cfg.vqgan_type == "objaworld":
                vqgan_codebook = nn.Embedding(1024, 256)
                vqgan_codebook.weight.data = torch.tensor(
                    np.load(self.OBJAWORLD_VQGAN_CODEBOOK_PATH)
                )
            elif cfg.vqgan_type == "objectron":
                vqgan_codebook = nn.Embedding(1024, 256)
                vqgan_codebook.weight.data = torch.tensor(
                    np.load(self.OBJECTRON_VQGAN_CODEBOOK_PATH)
                )

            ground_truth_image_tokens = []
            generated_image_tokens = []

            for token in generated_tokens[0][prompt.size(0) :]:
                if token >= cfg.image_token_offset:
                    generated_image_tokens.append(
                        token.cpu().numpy().item() - cfg.image_token_offset
                    )

            # for token in generated_tokens[0][:prompt.size(0)]:
            for token in groundtruth_target_tokens:
                if token >= cfg.image_token_offset:
                    ground_truth_image_tokens.append(token - cfg.image_token_offset)

            if len(only_generated_image_tokens) > 0:
                if cfg.reorder_image_tokens:
                    ground_truth_image_tokens = self.reverse_reordered_list(
                        ground_truth_image_tokens
                    )
                    if len(generated_image_tokens) == 0:
                        # make them all 0
                        generated_image_tokens = [0] * len(ground_truth_image_tokens)
                    else:
                        generated_image_tokens = self.reverse_reordered_list(
                            generated_image_tokens
                        )

                generated_image_tokens = vqgan_codebook(
                    torch.tensor(generated_image_tokens, dtype=torch.long)
                ).detach()
                ground_truth_image_tokens = vqgan_codebook(
                    torch.tensor(ground_truth_image_tokens, dtype=torch.long)
                ).detach()

                # define two tensors of shape 256, cfg.vqgan_row_col_size, cfg.vqgan_row_col_size and copy the values from the generated_image_tokens and ground_truth_image_tokens row major wise
                generated_embedding = torch.zeros(
                    256, cfg.vqgan_row_col_size, cfg.vqgan_row_col_size
                )
                ground_truth_embedding = torch.zeros(
                    256, cfg.vqgan_row_col_size, cfg.vqgan_row_col_size
                )

                try:
                    for i in range(cfg.vqgan_row_col_size):
                        for j in range(cfg.vqgan_row_col_size):
                            generated_embedding[:, i, j] = generated_image_tokens[
                                i * cfg.vqgan_row_col_size + j
                            ]
                            ground_truth_embedding[:, i, j] = ground_truth_image_tokens[
                                i * cfg.vqgan_row_col_size + j
                            ]
                except:
                    print("Index out of bounds!!!")
                    print("Number of tokens:", len(generated_image_tokens))

                # save as npy files
                # check if directory exists else create it
                if not os.path.exists(
                    f"./{cfg.image_embeddings_output_folder}/{cfg.run_identifier}"
                ):
                    os.makedirs(
                        f"./{cfg.image_embeddings_output_folder}/{cfg.run_identifier}"
                    )
                np.save(
                    f"./{cfg.image_embeddings_output_folder}/{cfg.run_identifier}/generated_image_tokens_{cfg.run_identifier}_{self.text_data_keys[sample_id]}_{sample_id}.npy",
                    generated_embedding.numpy(),
                )
                np.save(
                    f"./{cfg.image_embeddings_output_folder}/{cfg.run_identifier}/ground_truth_image_tokens_{cfg.run_identifier}_{self.text_data_keys[sample_id]}_{sample_id}.npy",
                    ground_truth_embedding.numpy(),
                )

            model_size = sum(
                [
                    p.numel() * p.dtype.itemsize
                    for p in itertools.chain(
                        self._model.parameters(), self._model.buffers()
                    )
                ]
            )

            num_tokens_generated = len(generated_tokens[0]) - prompt.size(0)

            tokens_sec = num_tokens_generated / t
            logger.info(f"Tokens generated: {num_tokens_generated}")
            logger.info(
                f"Time for inference: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec"
            )
            logger.info(
                f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s"
            )
            logger.info(
                f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB"
            )

        if not os.path.exists(
            f"./{cfg.three_d_json_output_folder}/{cfg.run_identifier}"
        ):
            os.makedirs(f"./{cfg.three_d_json_output_folder}/{cfg.run_identifier}")

        with open(
            f"./{cfg.three_d_json_output_folder}/{cfg.run_identifier}/groundtruth_scenes_{cfg.run_identifier}.json",
            "w",
        ) as f:
            json.dump({"scenes": groundtruth_scenes}, f, indent=4)

        with open(
            f"./{cfg.three_d_json_output_folder}/{cfg.run_identifier}/predicted_scenes_{cfg.run_identifier}.json",
            "w",
        ) as f:
            json.dump({"scenes": predicted_scenes}, f, indent=4)

        with open(
            f"./{cfg.three_d_json_output_folder}/{cfg.run_identifier}/original_scenes_{cfg.run_identifier}.json",
            "w",
        ) as f:
            json.dump({"scenes": original_scenes}, f, indent=4)

        if cfg.load_text_target_data:
            with open(
                f"./{cfg.three_d_json_output_folder}/{cfg.run_identifier}/groundtruth_answers_{cfg.run_identifier}.json",
                "w",
            ) as f:
                json.dump({"answers": groundtruth_answers}, f, indent=4)

            with open(
                f"./{cfg.three_d_json_output_folder}/{cfg.run_identifier}/predicted_answers_{cfg.run_identifier}.json",
                "w",
            ) as f:
                json.dump({"answers": predicted_answers}, f, indent=4)


@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="InferenceRecipe", cfg=cfg)
    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.generate(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
