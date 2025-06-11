# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Mapping, Optional

from torch.utils.data import Dataset
import json
import numpy as np
import math
from torchtune.modules.tokenizers import ModelTokenizer
from tqdm import tqdm


class ThreeDMLLMDataset(Dataset):
    """
    Dataset class for Kyvo data. This dataset is used for multimodal tasks that require text, image, and 3D data.

    Args:
        task_type (str): The type of task to perform. Options are "I-3", "3-I", "I+Q-A", "I+3+Q-A", "3+I+Q-A", "3+T-3", etc
        text_source (str): The path to the text source file.
        image_source (str): The path to the image source file.
        three_d_source (str): The path to the 3d source file.
        text_target (str): The path to the text target file.
        image_target (str): The path to the image target file.
        three_d_target (str): The path to the 3d target file.
        max_seq_len (int): The maximum sequence length to use for the model. Default is None.
        image_token_offset (int): The offset to add to the image tokens. Default is 0.
        load_dataset_kwargs (Dict[str, Any]): Additional keyword arguments to pass to the
            dataset loader.
    """

    def __init__(
        self,
        task_type: str,  # "I-3", "3-I", "I+Q-A", "I+3+Q-A", "3+I+Q-A", "3+T-3", etc
        text_source: str,  # json file path
        image_source: str = None,  # json file path
        three_d_source: str = None,  # json file path
        text_target: str = None,  # json file path
        image_target: str = None,  # json file path
        three_d_target: str = None,  # json file path
        max_seq_len: Optional[int] = None,
        image_token_offset: int = 129471,  # tokenizer dependent
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:

        self.task_type = task_type  # "vqa", "edit", "difference"
        if self.task_type == "UNIFIED":
            self.unified_task = True
        else:
            self.unified_task = False
        self.no_loss_on_input = load_dataset_kwargs.get("no_loss_on_input", False)
        self.reorder_image_tokens = load_dataset_kwargs.get(
            "reorder_image_tokens", False
        )
        self.image_token_offset = image_token_offset
        self.max_seq_len = max_seq_len
        self.num_samples = load_dataset_kwargs.get("num_samples", None)

        self.dataset_name = load_dataset_kwargs.get("dataset_name", None)
        assert self.dataset_name is not None, "Dataset name is required"

        self.load_text_source = load_dataset_kwargs.get("load_text_source", False)
        self.load_image_source = load_dataset_kwargs.get("load_image_source", False)
        self.load_three_d_source = load_dataset_kwargs.get("load_three_d_source", False)
        self.load_text_target = load_dataset_kwargs.get("load_text_target", False)
        self.load_image_target = load_dataset_kwargs.get("load_image_target", False)
        self.load_three_d_target = load_dataset_kwargs.get("load_three_d_target", False)

        self._text_data = None
        self._text_data_keys = None
        self._image_data = None
        self._image_data_keys = None
        self._three_d_data = None
        self._three_d_data_keys = None
        self._text_target = None
        self._text_target_keys = None
        self._image_target = None
        self._image_target_keys = None
        self._three_d_target = None
        self._three_d_target_keys = None

        if self.load_text_source:
            print("Loading text source")
            with open(text_source, "r") as f:
                text_data = json.load(f)
            print("Preparing text data")
            self._text_data = {
                k: v["token_ids"] for k, v in tqdm(sorted(text_data.items()))
            }
            self._text_data_keys = list(sorted(self._text_data.keys()))
            print("Number of text samples: ", len(self._text_data_keys))

        if self.load_image_source:
            print("Loading image source")
            with open(image_source, "r") as f:
                image_data = json.load(f)

            if (
                self.reorder_image_tokens
                and self.task_type != "I+3+T-I+3"
                and self.task_type != "3+I+T-3+I"
            ):
                print("Reordering image tokens (source)")
                self._image_data = {
                    k: self.reorder_list_optimized(image_data[k])
                    for k in tqdm(sorted(image_data.keys()))
                }
                self._image_data_keys = list(sorted(self._image_data.keys()))
            else:
                self._image_data = {
                    k: image_data[k] for k in tqdm(sorted(image_data.keys()))
                }
                self._image_data_keys = list(sorted(self._image_data.keys()))
            print("Number of image samples: ", len(self._image_data_keys))

        if self.load_three_d_source:
            print("Loading 3D source")
            with open(three_d_source, "r") as f:
                three_d_data = json.load(f)
            print("Preparing 3D data")
            self._three_d_data = {
                k: v["token_ids"] for k, v in tqdm(sorted(three_d_data.items()))
            }
            self._three_d_data_keys = list(sorted(self._three_d_data.keys()))
            print("Number of 3D samples: ", len(self._three_d_data_keys))

        if self.load_text_target:
            print("Loading text target")
            with open(text_target, "r") as f:
                text_target = json.load(f)
            self._text_target = {
                k: v["token_ids"] for k, v in sorted(text_target.items())
            }
            self._text_target_keys = list(sorted(self._text_target.keys()))
            print("Number of text target samples: ", len(self._text_target))

        if self.load_image_target:
            print("Loading image target")
            with open(image_target, "r") as f:
                image_target = json.load(f)
            if self.reorder_image_tokens:
                print("Reordering image tokens (target)")
                self._image_target = {
                    k: self.reorder_list_optimized(image_target[k])
                    for k in tqdm(sorted(image_target.keys()))
                }
                self._image_target_keys = list(sorted(self._image_target.keys()))
            else:
                self._image_target = {
                    k: image_target[k] for k in tqdm(sorted(image_target.keys()))
                }
                self._image_target_keys = list(sorted(self._image_target.keys()))
            print("Number of image target samples: ", len(self._image_target))

        if self.load_three_d_target:
            print("Loading 3D target")
            with open(three_d_target, "r") as f:
                three_d_target = json.load(f)
            self._three_d_target = {
                k: v["token_ids"] for k, v in sorted(three_d_target.items())
            }
            self._three_d_target_keys = list(sorted(self._three_d_target.keys()))
            print("Number of 3D target samples: ", len(self._three_d_target))

        # get the common keys for all the data which are not None
        common_keys = set()
        if self._text_data is not None:
            if len(common_keys) == 0:
                common_keys = set(self._text_data.keys())
            # common_keys = common_keys.union(set(self._text_data.keys()))
        if self._image_data is not None:
            if len(common_keys) == 0:
                common_keys = set(self._image_data.keys())
            common_keys = common_keys.intersection(set(self._image_data.keys()))
        if self._three_d_data is not None:
            if len(common_keys) == 0:
                common_keys = set(self._three_d_data.keys())
            common_keys = common_keys.intersection(set(self._three_d_data.keys()))
        if self._text_target is not None:
            if len(common_keys) == 0:
                common_keys = set(self._text_target.keys())
            common_keys = common_keys.intersection(set(self._text_target.keys()))
        if self._image_target is not None:
            if len(common_keys) == 0:
                common_keys = set(self._image_target.keys())
            common_keys = common_keys.intersection(set(self._image_target.keys()))
        if self._three_d_target is not None:
            if len(common_keys) == 0:
                common_keys = set(self._three_d_target.keys())
            common_keys = common_keys.intersection(set(self._three_d_target.keys()))

        if len(common_keys) == 0:
            raise ValueError("No common keys found for all the data")

        if self.num_samples is not None:
            common_keys = list(common_keys)[: self.num_samples]

        if self._text_data is not None:
            self._text_data = {k: self._text_data[k] for k in common_keys}
            self._text_data_keys = list(sorted(self._text_data.keys()))

        if self._image_data is not None:
            self._image_data = {k: self._image_data[k] for k in common_keys}
            self._image_data_keys = list(sorted(self._image_data.keys()))

        if self._three_d_data is not None:
            self._three_d_data = {k: self._three_d_data[k] for k in common_keys}
            self._three_d_data_keys = list(sorted(self._three_d_data.keys()))

        if self._text_target is not None:
            self._text_target = {k: self._text_target[k] for k in common_keys}
            self._text_target_keys = list(sorted(self._text_target.keys()))

        if self._image_target is not None:
            self._image_target = {k: self._image_target[k] for k in common_keys}
            self._image_target_keys = list(sorted(self._image_target.keys()))

        if self._three_d_target is not None:
            self._three_d_target = {k: self._three_d_target[k] for k in common_keys}
            self._three_d_target_keys = list(sorted(self._three_d_target.keys()))

        print("Total number of samples: ", len(common_keys))
        print(f"Initialization of Kyvo dataset complete for {self.dataset_name}!")

    def reorder_list_optimized(self, data):
        n = len(data)
        center = int((n // 2) - (math.sqrt(n) // 2))  # center of the image
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

    def __len__(self):
        try:
            return len(self._text_data)
        except:
            return len(self._three_d_data)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:

        # assert that the keys are the same for all the data both source and target which are not None
        assert_keys = []
        if self._text_data is not None:
            assert_keys.append(self._text_data_keys[index])
            text_sample = self._text_data[self._text_data_keys[index]]
        if self._image_data is not None:
            assert_keys.append(self._image_data_keys[index])
            image_sample = self._image_data[self._image_data_keys[index]]
            image_sample = [x + self.image_token_offset for x in image_sample]
        if self._three_d_data is not None:
            assert_keys.append(self._three_d_data_keys[index])
            three_d_sample = self._three_d_data[self._three_d_data_keys[index]]
        if self._text_target is not None:
            assert_keys.append(self._text_target_keys[index])
            text_target = self._text_target[self._text_target_keys[index]]
        if self._image_target is not None:
            assert_keys.append(self._image_target_keys[index])
            image_target = self._image_target[self._image_target_keys[index]]
            image_target = [x + self.image_token_offset for x in image_target]
        if self._three_d_target is not None:
            assert_keys.append(self._three_d_target_keys[index])
            three_d_target = self._three_d_target[self._three_d_target_keys[index]]

        # assert that the keys are the same for all the data both source and target
        assert all(x == assert_keys[0] for x in assert_keys), f"Keys are not the same"

        BOS = [128000]
        EOS = [128001]

        if self.dataset_name == "CLEVR" or self.dataset_name == "ObjaWorld":
            BOIMG = [129466]
            EOIMG = [129467]
            OUTSEP = [129470]
        elif self.dataset_name == "Omni3D-Objectron":
            BOIMG = [128366]
            EOIMG = [128367]
            OUTSEP = [128370]

        if self.unified_task:
            # choose a random task type
            task_types = [
                "3-I",
                # "I+Q-A",
                # "I+3+Q-A",
                "3+T-3",
                "I+3+T-3",
                "I+3+T-I+3",
                "I+T-I+3",
                "I+T-3",
                "I+T-I",
                "I-3",
            ]
            self.task_type = np.random.choice(task_types)

        if self.task_type == "3-I":
            # 3D --> Image
            tokens = BOS + three_d_sample + OUTSEP + BOIMG + image_sample + EOIMG + EOS
            if self.no_loss_on_input:
                labels = (
                    [-100] * len(BOS + three_d_sample + OUTSEP)
                    + BOIMG
                    + image_sample
                    + EOIMG
                    + EOS
                )
            else:
                labels = tokens.copy()

        elif self.task_type == "I+Q-A":
            # 3D --> Image
            tokens = (
                BOS
                + BOIMG
                + image_sample
                + EOIMG
                + text_sample
                + OUTSEP
                + text_target
                + EOS
            )
            if self.no_loss_on_input:
                labels = (
                    [-100]
                    * len(BOS + BOIMG + image_sample + EOIMG + text_sample + OUTSEP)
                    + text_target
                    + EOS
                )
            else:
                labels = tokens.copy()

        elif self.task_type == "I+3+Q-A":
            # 3D --> Image
            tokens = (
                BOS
                + BOIMG
                + image_sample
                + EOIMG
                + three_d_sample
                + text_sample
                + OUTSEP
                + text_target
                + EOS
            )
            if self.no_loss_on_input:
                labels = (
                    [-100]
                    * len(
                        BOS
                        + BOIMG
                        + image_sample
                        + EOIMG
                        + three_d_sample
                        + text_sample
                        + OUTSEP
                    )
                    + text_target
                    + EOS
                )
            else:
                labels = tokens.copy()

        elif self.task_type == "3+I+Q-A":
            # 3D --> Image
            tokens = (
                BOS
                + three_d_sample
                + BOIMG
                + image_sample
                + EOIMG
                + text_sample
                + OUTSEP
                + text_target
                + EOS
            )
            if self.no_loss_on_input:
                labels = (
                    [-100]
                    * len(
                        BOS
                        + three_d_sample
                        + BOIMG
                        + image_sample
                        + EOIMG
                        + text_sample
                        + OUTSEP
                    )
                    + text_target
                    + EOS
                )
            else:
                labels = tokens.copy()

        elif self.task_type == "3-I-first3output":
            # 3D --> Image
            tokens = BOS + three_d_sample + OUTSEP + BOIMG + image_sample + EOIMG + EOS
            if self.no_loss_on_input:
                labels = (
                    [-100]
                    * len(BOS + three_d_sample + OUTSEP + BOIMG + image_sample[:3])
                    + image_sample[3:]
                    + EOIMG
                    + EOS
                )
            else:
                labels = tokens.copy()

        elif self.task_type == "3+T-3":
            # 3D + Text --> 3D
            tokens = BOS + three_d_sample + text_sample + OUTSEP + three_d_target + EOS
            if self.no_loss_on_input:
                labels = (
                    [-100] * len(BOS + three_d_sample + text_sample + OUTSEP)
                    + three_d_target
                    + EOS
                )
            else:
                labels = tokens.copy()

        elif self.task_type == "I+3+T-3":
            # Image + 3D + Text --> 3D
            tokens = (
                BOS
                + BOIMG
                + image_sample
                + EOIMG
                + three_d_sample
                + text_sample
                + OUTSEP
                + three_d_target
                + EOS
            )
            if self.no_loss_on_input:
                labels = (
                    [-100]
                    * len(
                        BOS
                        + BOIMG
                        + image_sample
                        + EOIMG
                        + three_d_sample
                        + text_sample
                        + OUTSEP
                    )
                    + three_d_target
                    + EOS
                )
            else:
                labels = tokens.copy()

        elif self.task_type == "I+3+T-I+3":
            # Image + 3D + Text --> Image + 3D
            tokens = (
                BOS
                + BOIMG
                + image_sample
                + EOIMG
                + three_d_sample
                + text_sample
                + OUTSEP
                + BOIMG
                + image_target
                + EOIMG
                + three_d_target
                + EOS
            )
            if self.no_loss_on_input:
                labels = (
                    [-100]
                    * len(
                        BOS
                        + BOIMG
                        + image_sample
                        + EOIMG
                        + three_d_sample
                        + text_sample
                        + OUTSEP
                    )
                    + BOIMG
                    + image_target
                    + EOIMG
                    + three_d_target
                    + EOS
                )
            else:
                labels = tokens.copy()

        elif self.task_type == "3+I+T-3+I":
            # 3D + Image + Text --> 3D + Image
            tokens = (
                BOS
                + three_d_sample
                + BOIMG
                + image_sample
                + EOIMG
                + text_sample
                + OUTSEP
                + three_d_target
                + BOIMG
                + image_target
                + EOIMG
                + EOS
            )
            if self.no_loss_on_input:
                labels = (
                    [-100]
                    * len(
                        BOS
                        + three_d_sample
                        + BOIMG
                        + image_sample
                        + EOIMG
                        + text_sample
                        + OUTSEP
                    )
                    + three_d_target
                    + BOIMG
                    + image_target
                    + EOIMG
                    + EOS
                )
            else:
                labels = tokens.copy()

        elif self.task_type == "I+T-I+3":
            # Image + Text --> Image + 3D
            tokens = (
                BOS
                + BOIMG
                + image_sample
                + EOIMG
                + text_sample
                + OUTSEP
                + BOIMG
                + image_target
                + EOIMG
                + three_d_target
                + EOS
            )
            if self.no_loss_on_input:
                labels = (
                    [-100]
                    * len(BOS + BOIMG + image_sample + EOIMG + text_sample + OUTSEP)
                    + BOIMG
                    + image_target
                    + EOIMG
                    + three_d_target
                    + EOS
                )
            else:
                labels = tokens.copy()

        elif self.task_type == "I+T-3":
            # Image + Text --> 3D
            tokens = (
                BOS
                + BOIMG
                + image_sample
                + EOIMG
                + text_sample
                + OUTSEP
                + three_d_target
                + EOS
            )
            if self.no_loss_on_input:
                labels = (
                    [-100]
                    * len(BOS + BOIMG + image_sample + EOIMG + text_sample + OUTSEP)
                    + three_d_target
                    + EOS
                )
            else:
                labels = tokens.copy()

        elif self.task_type == "I+T-I":
            # Image + Text --> Image
            tokens = (
                BOS
                + BOIMG
                + image_sample
                + EOIMG
                + text_sample
                + OUTSEP
                + BOIMG
                + image_target
                + EOIMG
                + EOS
            )
            if self.no_loss_on_input:
                labels = (
                    [-100]
                    * len(BOS + BOIMG + image_sample + EOIMG + text_sample + OUTSEP)
                    + BOIMG
                    + image_target
                    + EOIMG
                    + EOS
                )
            else:
                labels = tokens.copy()

        elif self.task_type == "I-3":
            # Image --> 3D
            tokens = BOS + BOIMG + image_sample + EOIMG + OUTSEP + three_d_sample + EOS
            if self.no_loss_on_input:
                labels = (
                    [-100] * len(BOS + BOIMG + image_sample + EOIMG + OUTSEP)
                    + three_d_sample
                    + EOS
                )
            else:
                labels = tokens.copy()

        elif self.task_type == "I+T-T":
            # Image + Text --> Text
            tokens = (
                BOS
                + BOIMG
                + image_sample
                + EOIMG
                + text_sample
                + OUTSEP
                + text_target
                + EOS
            )
            if self.no_loss_on_input:
                labels = (
                    [-100]
                    * len(BOS + BOIMG + image_sample + EOIMG + text_sample + OUTSEP)
                    + text_target
                    + EOS
                )
            else:
                labels = tokens.copy()

        elif self.task_type == "I+3+T-T":
            # Image + 3D + Text --> Text
            tokens = (
                BOS
                + BOIMG
                + image_sample
                + EOIMG
                + three_d_sample
                + text_sample
                + OUTSEP
                + text_target
                + EOS
            )
            if self.no_loss_on_input:
                labels = (
                    [-100]
                    * len(
                        BOS
                        + BOIMG
                        + image_sample
                        + EOIMG
                        + three_d_sample
                        + text_sample
                        + OUTSEP
                    )
                    + text_target
                    + EOS
                )
            else:
                labels = tokens.copy()

        else:
            raise ValueError(f"Task type {self.task_type} not recognized")

        return {
            "tokens": tokens,
            "labels": labels,
        }


def threed_mllm_dataset(
    task_type: str,
    text_source: str,
    image_source: str,
    three_d_source: str,
    max_seq_len: Optional[int] = None,
    image_token_offset: int = 0,
    **load_dataset_kwargs: Dict[str, Any],
) -> ThreeDMLLMDataset:
    return ThreeDMLLMDataset(
        task_type=task_type,
        text_source=text_source,
        image_source=image_source,
        three_d_source=three_d_source,
        max_seq_len=max_seq_len,
        image_token_offset=image_token_offset,
        **load_dataset_kwargs,
    )
