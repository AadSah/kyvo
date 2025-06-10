import torch
import yaml
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel, GumbelVQ
import io
import requests
import PIL
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config


def load_vqgan(config, ckpt_path=None, is_gumbel=False):
    if is_gumbel:
        model = GumbelVQ(**config.model.params)
    else:
        model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()


def preprocess_vqgan(x):
    x = 2.0 * x - 1.0
    return x


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def reconstruct_with_vqgan(x, model, reconstruct=False):
    # could also use model(x) for reconstruction but use explicit encoding and decoding here
    z, _, [_, _, indices] = model.encode(x)
    if reconstruct:
        xrec = model.decode(z)
        return xrec, indices
    else:
        return z, indices


def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return PIL.Image.open(io.BytesIO(resp.content))


def get_local_image(path, make_square=True, size=320, horizontal_flip=False):
    img = PIL.Image.open(path)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    if make_square:
        img = img.resize((size, size))
    if horizontal_flip:
        img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    return img


def preprocess(img, target_image_size=320):
    s = min(img.size)

    if s < target_image_size:
        raise ValueError(f"min dim for image {s} < {target_image_size}")

    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)

    return img


def reconstruction_pipeline(
    path,
    vqgan_model,
    size=320,
    DEVICE="cuda:0",
    reconstruct=False,
    make_square=True,
    horizontal_flip=False,
):
    x_vqgan = preprocess(
        get_local_image(
            path, make_square=make_square, size=size, horizontal_flip=horizontal_flip
        ),
        target_image_size=size,
    )
    x_vqgan = x_vqgan.to(DEVICE)
    vqgan_embedding, vqgan_indices = reconstruct_with_vqgan(
        preprocess_vqgan(x_vqgan), vqgan_model, reconstruct=reconstruct
    )
    return vqgan_embedding, vqgan_indices
