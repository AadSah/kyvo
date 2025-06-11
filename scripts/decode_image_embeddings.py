import torch
import yaml
from omegaconf import OmegaConf
import argparse
import os
import torch
from tqdm import tqdm
import json
import numpy as np

from taming_transformers_utils import *

import argparse

parser = argparse.ArgumentParser(description="Decode embeddings to images")

parser.add_argument(
    "--folder_path",
    required=True,
    type=str,
    help="Path to the folder containing the embeddings",
)

parser.add_argument(
    "--vqgan_type",
    required=True,
    type=str,
    help="Type of VQGAN model used for training: choose from [clevr, objaworld, objectron, domain-agnostic]",
)

parser.add_argument(
    "--image_output_path",
    required=True,
    type=str,
    help="Path to the folder where the decoded images will be saved",
)

args = parser.parse_args()

# add arguments
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_grad_enabled(False)

is_gumbel = False
if args.vqgan_type == "clevr":
    config_path = "./kyvo-datasets-and-codebooks/vqgan-models-and-codebooks/clevr/2024-10-10T09-21-36_custom_vqgan_CLEVR-LARGE/configs/2024-10-10T09-21-36-project.yaml"
    ckpt_path = "./kyvo-datasets-and-codebooks/vqgan-models-and-codebooks/clevr/2024-10-10T09-21-36_custom_vqgan_CLEVR-LARGE/checkpoints/last.ckpt"
elif args.vqgan_type == "objaworld":
    config_path = "./kyvo-datasets-and-codebooks/vqgan-models-and-codebooks/objaworld/2025-01-17T09-02-22_custom_vqgan_SYNTHETIC_LIVINGROOM_PARK_LARGE_EP100/configs/2025-01-17T09-02-22-project.yaml"
    ckpt_path = "./kyvo-datasets-and-codebooks/vqgan-models-and-codebooks/objaworld/2025-01-17T09-02-22_custom_vqgan_SYNTHETIC_LIVINGROOM_PARK_LARGE_EP100/checkpoints/last.ckpt"
elif args.vqgan_type == "objectron":
    config_path = "./kyvo-datasets-and-codebooks/vqgan-models-and-codebooks/objectron/2024-11-03T05-41-42_custom_vqgan_OMNI3D_OBJECTRON_ep200/configs/2024-11-03T05-41-42-project.yaml"
    ckpt_path = "./kyvo-datasets-and-codebooks/vqgan-models-and-codebooks/objectron/2024-11-03T05-41-42_custom_vqgan_OMNI3D_OBJECTRON_ep200/checkpoints/last.ckpt"
elif args.vqgan_type == "domain-agnostic":
    config_path = "./kyvo-datasets-and-codebooks/vqgan-models-and-codebooks/domain-agnostic/vqgan_gumbel_f8/configs/model.yaml"
    ckpt_path = "./kyvo-datasets-and-codebooks/vqgan-models-and-codebooks/domain-agnostic/vqgan_gumbel_f8/checkpoints/last.ckpt"
    is_gumbel = True

config = load_config(
    config_path,
    display=False,
)
model = load_vqgan(
    config,
    ckpt_path=ckpt_path,
    is_gumbel=is_gumbel,
).to(DEVICE)


folder_path = args.folder_path

groundtruth_names = [
    f for f in os.listdir(folder_path) if "ground_truth" in f and f.endswith(".npy")
]
generated_names = [
    f for f in os.listdir(folder_path) if "generated" in f and f.endswith(".npy")
]

# natural sort
groundtruth_names.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
generated_names.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

folder_identifier = os.path.basename(folder_path)
groundtruth_key_names = []
for name in groundtruth_names:
    key = name.split(".png")[0].split(folder_identifier)[-1]
    groundtruth_key_names.append(str(key))

generated_key_names = []
for name in generated_names:
    key = name.split(".png")[0].split(folder_identifier)[-1]
    generated_key_names.append(str(key))

assert len(groundtruth_names) == len(generated_names)
print("Total number of embeddings to decode: ", len(groundtruth_names))

for img_idx in tqdm(range(len(groundtruth_names)), desc="Decoding images"):
    # load embeddings from npy files and convert to torch tensors
    groundtruth_embeddings = torch.tensor(
        np.load(os.path.join(folder_path, groundtruth_names[img_idx]))
    ).to(DEVICE)
    generated_embeddings = torch.tensor(
        np.load(os.path.join(folder_path, generated_names[img_idx]))
    ).to(DEVICE)

    xrec_groundtruth = model.decode(groundtruth_embeddings.unsqueeze(0))
    xrec_generated = model.decode(generated_embeddings.unsqueeze(0))

    if not os.path.exists(os.path.join(args.image_output_path, "GROUNDTRUTH")):
        os.makedirs(os.path.join(args.image_output_path, "GROUNDTRUTH"))

    if not os.path.exists(os.path.join(args.image_output_path, "GENERATED")):
        os.makedirs(os.path.join(args.image_output_path, "GENERATED"))

    custom_to_pil(xrec_groundtruth[0]).save(
        os.path.join(
            args.image_output_path,
            "GROUNDTRUTH",
            f"ground_truth{groundtruth_key_names[img_idx]}.png",
        )
    )
    custom_to_pil(xrec_generated[0]).save(
        os.path.join(
            args.image_output_path,
            "GENERATED",
            f"generated{generated_key_names[img_idx]}.png",
        )
    )
