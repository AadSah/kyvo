# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import torch
from torchtune import training

import checkpointer.convert_weights as convert_weights
from torchtune.training.checkpointing._utils import (
    get_path,
    ModelType,
    safe_torch_load,
)
from torchtune.utils._logging import get_logger

logger = get_logger("DEBUG")


class _CheckpointerInterface(Protocol):
    """
    Interface implemented by Checkpointers in torchtune.

    torchtune checkpointers are designed to be composable components which can be plugged
    into any training recipe. Each checkpointer supports a specific set of models and training
    scenarios making these easy to understand, debug and extend. For example, the
    ``FullModelCheckpointer``s are used for loading and saving all of the model weights.
    This checkpointer can be used for Full-Finetuning scenarios or PEFT where the output is a
    merged checkpoint. In case the current suite of checkpointers are inadequate,
    users are encouraged to implement their own and contribute back to torchtune.

    torchtune is also designed to be "state-dict invariant". This means the checkpointer
    ensures that the output checkpoint has the same format as the original checkpoint i.e.
    the output checkpoint has the same keys split across the same number of files as the original
    checkpoint. Being "state-dict invariant" allows users to seamlessly use torchtune checkpoints
    with their favorite post-training tools from the open-source ecosystem without writing
    torchtune-specific convertors. To be "state-dict invariant", the ``load_checkpoint`` and
    ``save_checkpoint`` methods make use of the weight convertors available in
    ``torchtune/models/<model_folder>``.

    torchtune Checkpointers support two checkpointing scenarios:
        * End-of-training Checkpointing. The model weights at the end of a completed training
            run are written out to file. The checkpointer ensures that the output checkpoint
            files have the same keys as the input checkpoint file used to begin training. The
            checkpointer also ensures that the keys are partitioned across the same number of
            files as the original checkpoint. This ensures that the original metadata files can
            be used as is, and the output checkpoint can be used with any tool that understands
            the original checkpoint format. This includes popular inference engines such as
            ``llama.cpp`` and ``gpt-fast``. The output state dict has the following format:
            {
                "key_1": weight
                ...
            }


        Mid-training Chekpointing. In addition to the model checkpoint files, we output an
            additional "recipe_state.pt" file for intermediate checkpoints. These are currently
            output at the end of each epoch, and contain information such as optimizer state,
            number of epochs completed etc which is needed to correctly resume a previously
            interrupted training run. The recipe is responsible for constructing the state dict
            with the information it needs. The checkpointer extracts the model state dict
            (key = "model") and writes everything else out to "recipe_state.pt". To prevent us
            from flooding ``output_dir`` with checkpoint files, the recipe state is overwritten
            at the end of each epoch. The output state dicts have the following formats:

            Model:
                {
                    "key_1": weight
                    ...
                }

            Recipe State:
                {
                    "optimizer": ...,
                    "epoch": ...,
                    ...
                }

    """

    def load_checkpoint(self, **kwargs) -> Dict[str, Any]: ...

    def save_checkpoint(self, state_dict: Dict[str, Any], **kwargs) -> None: ...


class FullModelMetaCheckpointer3D(_CheckpointerInterface):
    """
    Checkpointer which reads and writes checkpoints in Meta's format. Examples include
    the Llama-2-7b model from the meta-llama repo (https://huggingface.co/meta-llama/Llama-2-7b)

    Currently we support reading from a single checkpoint file only. Support for reading from
    sharded checkpoints is WIP.

    Args:
        checkpoint_dir (str): Directory containing the checkpoint files
        checkpoint_files (List[str]): List of checkpoint files to load. Currently this checkpointer only
            supports loading a single checkpoint file.
        model_type (ModelType): Model type of the model for which the checkpointer is being loaded
        output_dir (str): Directory to save the checkpoint files
        adapter_checkpoint (Optional[str]): Path to the adapter weights. Default is None
        recipe_checkpoint (Optional[str]): Path to the recipe state checkpoint file. Default is None
        resume_from_checkpoint (bool): If True, the checkpointer will load the additional checkpoint files to
            resume training from a previous run. Default is False

    Raises:
        ValueError: If ``checkpoint_files`` is not a list of length 1
        ValueError: If ``resume_from_checkpoint`` is True but ``recipe_checkpoint`` is None
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_files: List[str],
        model_type: ModelType,
        output_dir: str,
        adapter_checkpoint: Optional[str] = None,
        recipe_checkpoint: Optional[str] = None,
        resume_from_checkpoint: bool = False,
        convert_weights_type: str = "meta_to_tune_3d",
    ) -> None:
        # Fail fast if ``checkpoint_files`` is invalid
        if len(checkpoint_files) != 1:
            raise ValueError(
                "Currently we only support reading from a single torchtune checkpoint file. "
                f"Got {len(checkpoint_files)} files instead."
            )

        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_path = get_path(self._checkpoint_dir, checkpoint_files[0])

        self._adapter_checkpoint = (
            get_path(self._checkpoint_dir, adapter_checkpoint)
            if adapter_checkpoint
            else None
        )

        self._resume_from_checkpoint = resume_from_checkpoint
        self._model_type = ModelType[model_type]
        self._output_dir = Path(output_dir)

        # recipe_checkpoint contains the recipe state. This should be available if
        # resume_from_checkpoint is True
        self._recipe_checkpoint = None
        if self._resume_from_checkpoint:
            if recipe_checkpoint is None:
                raise ValueError(
                    "If resume_from_checkpoint is True, recipe_checkpoint file must be provided."
                )
            self._recipe_checkpoint = get_path(self._checkpoint_dir, recipe_checkpoint)

        self._convert_weights_type = convert_weights_type

    def load_checkpoint(self) -> Dict[str, Any]:
        """
        Load Meta checkpoint from file. Currently only loading from a single file is supported.
        """
        state_dict: Dict[str:Any] = {}
        model_state_dict = safe_torch_load(self._checkpoint_path)
        if self._model_type == ModelType.LLAMA3_VISION:
            from torchtune.models.llama3_2_vision._convert_weights import (
                llama3_vision_meta_to_tune,
            )

            state_dict[training.MODEL_KEY] = llama3_vision_meta_to_tune(
                model_state_dict
            )
        else:
            state_dict[training.MODEL_KEY] = convert_weights.meta_to_tune(
                model_state_dict
            )

        # llama3_2 has tied weights, so we need to remove the output.weight key
        if self._model_type == ModelType.LLAMA3_2:
            logger.info(
                "Identified model_type = Llama3_2. Ignoring output.weight in"
                " checkpoint in favor of the tok_embedding.weight"
                " tied weights."
            )
            state_dict[training.MODEL_KEY].pop("output.weight")

        if self._adapter_checkpoint:
            adapter_state_dict = safe_torch_load(self._adapter_checkpoint)
            state_dict[training.ADAPTER_KEY] = adapter_state_dict

        if self._resume_from_checkpoint:
            recipe_state = safe_torch_load(self._recipe_checkpoint, mmap=False)
            state_dict.update(recipe_state)
        return state_dict

    def load_checkpoint_3d(self) -> Dict[str, Any]:
        """
        Load Meta checkpoint from file. Currently only loading from a single file is supported.
        """
        state_dict: Dict[str:Any] = {}
        model_state_dict = safe_torch_load(self._checkpoint_path)
        if self._model_type == ModelType.LLAMA3_VISION:
            from torchtune.models.llama3_2_vision._convert_weights import (
                llama3_vision_meta_to_tune,
            )

            state_dict[training.MODEL_KEY] = llama3_vision_meta_to_tune(
                model_state_dict
            )
        else:
            state_dict[training.MODEL_KEY] = convert_weights.meta_to_tune_3d(
                model_state_dict, self._convert_weights_type
            )

        # llama3_2 has tied weights, so we need to remove the output.weight key
        if self._model_type == ModelType.LLAMA3_2:
            logger.info(
                "Identified model_type = Llama3_2. Ignoring output.weight in"
                " checkpoint in favor of the tok_embedding.weight"
                " tied weights."
            )
            # state_dict[training.MODEL_KEY].pop("output.weight")
            # We dont need to do this for 3D models

        if self._adapter_checkpoint:
            adapter_state_dict = safe_torch_load(self._adapter_checkpoint)
            state_dict[training.ADAPTER_KEY] = adapter_state_dict

        if self._resume_from_checkpoint:
            recipe_state = safe_torch_load(self._recipe_checkpoint, mmap=False)
            state_dict.update(recipe_state)
        return state_dict

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        epoch: int,
        intermediate_checkpoint: bool = False,
        adapter_only: bool = False,
    ) -> None:
        """
        Save Meta checkpoint to file. If ``intermediate_checkpoint`` is True, an additional
        checkpoint file ``recipe_state.pt`` is created in ``_output_dir`` which contains the recipe
        state.

        Args:
            state_dict (Dict[str, Any]): Checkpoint state dict to be written out to file
            epoch (int): Epoch number. Used to create the checkpoint file name
            intermediate_checkpoint (bool): If True, an additional checkpoint files for recipe state
                and (if applicable) adapter weights are created. Default is False
            adapter_only (bool): If True, only save the adapter weights. Default is False

        Raises:
            ValueError: if ``adapter_only`` is True and adapter checkpoint not found in state_dict.
        """
        self._output_dir.mkdir(exist_ok=True)

        if not adapter_only:
            model_state_dict = state_dict[training.MODEL_KEY]
            if self._model_type == ModelType.LLAMA3_VISION:
                from torchtune.models.llama3_2_vision._convert_weights import (
                    llama3_vision_tune_to_meta,
                )

                state_dict[training.MODEL_KEY] = llama3_vision_tune_to_meta(
                    model_state_dict
                )
            else:
                # llama3_2 has tied weights, so we need to add the output.weight key
                if (
                    self._model_type == ModelType.LLAMA3_2
                    and "output.weight" not in model_state_dict
                ):
                    model_state_dict["output.weight"] = model_state_dict[
                        "tok_embeddings.weight"
                    ]

                state_dict[training.MODEL_KEY] = convert_weights.tune_to_meta_3d(
                    model_state_dict, self._convert_weights_type
                )

            # Output file is always a .pt file with the epoch number in the name
            checkpoint_file = Path.joinpath(
                self._output_dir, f"meta_model_{epoch}"
            ).with_suffix(".pt")
            torch.save(state_dict[training.MODEL_KEY], checkpoint_file)
            logger.info(
                "Model checkpoint of size "
                f"{os.path.getsize(checkpoint_file) / 1000**3:.2f} GB "
                f"saved to {checkpoint_file}"
            )

        if training.ADAPTER_KEY in state_dict:
            output_path = Path.joinpath(
                self._output_dir, f"adapter_{epoch}"
            ).with_suffix(".pt")
            torch.save(state_dict[training.ADAPTER_KEY], output_path)
            logger.info(
                "Adapter checkpoint of size "
                f"{os.path.getsize(output_path) / 1000**3:.2f} GB "
                f"saved to {output_path}"
            )
        elif adapter_only:
            raise ValueError(
                "Adapter checkpoint not found in state_dict. Please ensure that the state_dict contains adapter weights."
            )

        # If the recipe state needs to be output, first remove the model state dict
        # and if it exists, remove the adapter state dict as well
        if intermediate_checkpoint:
            _ = state_dict.pop(training.MODEL_KEY)
            _ = state_dict.pop(training.ADAPTER_KEY, None)
            _ = state_dict.pop(training.ADAPTER_CONFIG, None)
            output_path = Path.joinpath(self._output_dir, "recipe_state.pt")
            torch.save(state_dict, output_path)
            logger.info(
                "Recipe checkpoint of size "
                f"{os.path.getsize(output_path) / 1000**3:.2f} GB "
                f"saved to {output_path}"
            )
        else:
            logger.info("Saving final epoch checkpoint.")
            if adapter_only:
                logger.info(
                    "Please note that you have set adapter_only=True, so only adapter weights will be saved."
                    "You need to merge the adapter weights into your base model for further use. "
                    f"See {self.__class__.__name__}.save_checkpoint for more details."
                )
            else:
                logger.info(
                    "The full model checkpoint, including all weights and configurations, has been saved successfully."
                    "You can now use this checkpoint for further training or inference."
                )
