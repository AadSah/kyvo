from transformers import AutoTokenizer
import numpy as np


def get_tokenizer(model_name="meta-llama/Llama-3.2-1B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    special_tokens_dict = {
        "additional_special_tokens": [
            "[SCENE-START]",
            "[SCENE-END]",
            "[OBJECT-START]",
            "[OBJECT-END]",
            "[SIZE]",
            "[COLOR]",
            "[MATERIAL]",
            "[SHAPE]",
            "[LOCATION]",
            "[IMAGE-START]",
            "[IMAGE-END]",
            "[TEXT-START]",
            "[TEXT-END]",
            "[OUTPUT-START]",
        ]
    }

    number_tokens = np.arange(-3, 3 + 0.005, 0.005).tolist()
    # convert the number tokens to strings
    number_tokens = [str(format(round(token, 3), ".3f")) for token in number_tokens]
    # replace "-0.000" with "0.000" in number tokens and update the list
    number_tokens = [token.replace("-0.000", "0.000") for token in number_tokens]

    tokenizer.add_tokens(number_tokens)
    tokenizer.add_special_tokens(special_tokens_dict)

    return tokenizer


def get_tokenizer_omni3d_objectron(model_name="meta-llama/Llama-3.2-1B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    special_tokens_dict = {
        "additional_special_tokens": [
            "[SCENE-START]",
            "[SCENE-END]",
            "[OBJECT-START]",
            "[OBJECT-END]",
            "[CATEGORY]",
            "[CENTER_CAM]",
            "[DIMENSIONS]",
            "[IMAGE-START]",
            "[IMAGE-END]",
            "[TEXT-START]",
            "[TEXT-END]",
            "[OUTPUT-START]",
        ]
    }

    center_cam_x_number_tokens = [0.01 * i for i in range(-20, 21)]
    center_cam_y_number_tokens = [0.01 * i for i in range(-20, 11)]
    center_cam_z_number_tokens = [0.05 * i for i in range(61)]
    center_cam_z_number_tokens.extend([5.0, 7.5, 10.0, 12.5, 15.0, 17.5])

    dimensions_length_number_tokens = [0.05 * i for i in range(21)]
    dimensions_width_number_tokens = [0.05 * i for i in range(21)]
    dimensions_height_number_tokens = [0.05 * i for i in range(25)]

    # merge, unique and sort the number tokens
    number_tokens = list(
        set(
            center_cam_x_number_tokens
            + center_cam_y_number_tokens
            + center_cam_z_number_tokens
            + dimensions_length_number_tokens
            + dimensions_width_number_tokens
            + dimensions_height_number_tokens
        )
    )
    number_tokens.sort()
    # convert the number tokens to strings
    number_tokens = [str(format(round(token, 3), ".2f")) for token in number_tokens]

    tokenizer.add_tokens(number_tokens)
    tokenizer.add_special_tokens(special_tokens_dict)

    return tokenizer
