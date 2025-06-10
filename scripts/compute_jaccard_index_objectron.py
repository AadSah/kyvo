import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--tau", type=float, default=0.25)
parser.add_argument("--dimensions_tau", type=float, default=0.05)
parser.add_argument("--predictions_file", type=str, required=True)
parser.add_argument("--groundtruth_file", type=str, required=True)
parser.add_argument(
    "--method",
    type=str,
    required=True,
    help="Method used for predictions - 'cube-rcnn' or '3d-mllm'",
)

args = parser.parse_args()


# load the file with all data
with open(args.groundtruth_file) as f:
    data = json.load(f)

# load the prediction json
with open(
    args.predictions_file,
) as f:
    predictions = json.load(f)

# preprocess the data
image_id_to_objects = {}
for ann in data["annotations"]:
    image_id = ann["image_id"]
    if image_id not in image_id_to_objects:
        image_id_to_objects[image_id] = {"objects": []}
    image_id_to_objects[image_id]["objects"].append(
        {
            "category_id": ann["category_id"],
            "category": ann["category_name"],
            "center_cam": ann["center_cam"],
            "dimensions": ann["dimensions"],
        }
    )


data_filepath_to_image_id = {}
for image in data["images"]:
    if image["file_path"] in data_filepath_to_image_id:
        raise ValueError("Duplicate file path found")
    data_filepath_to_image_id[image["file_path"]] = image["id"]


final_preprocessed_data = {}
for image_id, objects in image_id_to_objects.items():
    if image_id not in final_preprocessed_data:
        final_preprocessed_data[image_id] = {}
    for obj in objects["objects"]:
        if obj["category"] not in final_preprocessed_data[image_id]:
            final_preprocessed_data[image_id][obj["category"]] = {
                "num": 0,
                "center_cam_3d": [],
                "dimensions": [],
            }
        final_preprocessed_data[image_id][obj["category"]]["num"] += 1
        final_preprocessed_data[image_id][obj["category"]]["center_cam_3d"].append(
            np.array(obj["center_cam"])
        )
        final_preprocessed_data[image_id][obj["category"]]["dimensions"].append(
            np.array(obj["dimensions"])
        )

data_category_id_to_name = {}
for cat in data["categories"]:
    data_category_id_to_name[cat["id"]] = cat["name"]


if args.method == "cube-rcnn":
    cube_rcnn_predictions_preprocessed = {}
    for pred in predictions:
        image_id = pred["image_id"]
        if image_id not in cube_rcnn_predictions_preprocessed:
            cube_rcnn_predictions_preprocessed[image_id] = {"objects": []}
        cube_rcnn_predictions_preprocessed[image_id]["objects"].append(
            {
                "category_id": pred["category_id"],
                "category": data_category_id_to_name[pred["category_id"]],
                "center_cam": pred["center_cam"],
                "dimensions": pred["dimensions"],
            }
        )

    final_preprocessed_prediction_data = {}
    for image_id, objects in cube_rcnn_predictions_preprocessed.items():
        if image_id not in final_preprocessed_prediction_data:
            final_preprocessed_prediction_data[image_id] = {}
        for obj in objects["objects"]:
            if obj["category"] not in final_preprocessed_prediction_data[image_id]:
                final_preprocessed_prediction_data[image_id][obj["category"]] = {
                    "num": 0,
                    "center_cam_3d": [],
                    "dimensions": [],
                }
            final_preprocessed_prediction_data[image_id][obj["category"]]["num"] += 1
            final_preprocessed_prediction_data[image_id][obj["category"]][
                "center_cam_3d"
            ].append(np.array(obj["center_cam"]))
            final_preprocessed_prediction_data[image_id][obj["category"]][
                "dimensions"
            ].append(np.array(obj["dimensions"]))
elif args.method == "3d-mllm":
    final_preprocessed_prediction_data = {}
    for pred_scene in predictions["scenes"]:
        pred_objects = pred_scene["objects"]
        image_id = data_filepath_to_image_id[pred_scene["key"]]
        if image_id not in final_preprocessed_prediction_data:
            final_preprocessed_prediction_data[image_id] = {}
        for obj in pred_objects:
            if obj["category"] not in final_preprocessed_prediction_data[image_id]:
                final_preprocessed_prediction_data[image_id][obj["category"]] = {
                    "num": 0,
                    "center_cam_3d": [],
                    "dimensions": [],
                }
            final_preprocessed_prediction_data[image_id][obj["category"]]["num"] += 1
            final_preprocessed_prediction_data[image_id][obj["category"]][
                "center_cam_3d"
            ].append(np.array(obj["center_cam"]))
            final_preprocessed_prediction_data[image_id][obj["category"]][
                "dimensions"
            ].append(np.array(obj["dimensions"]))

# only keep the common keys
common_keys = set(final_preprocessed_data.keys()).intersection(
    set(final_preprocessed_prediction_data.keys())
)

final_preprocessed_data_common = {k: final_preprocessed_data[k] for k in common_keys}

final_preprocessed_prediction_data_common = {
    k: final_preprocessed_prediction_data[k] for k in common_keys
}

print("Total common keys: ", len(final_preprocessed_data_common.keys()))


import numpy as np
from scipy.spatial.distance import euclidean


def compute_jaccard(predictions, groundtruths, tau, dimensions_tau):
    """
    Compute the Jaccard metric based on predictions and ground truths.

    Parameters:
        predictions (list of dict): List of dictionaries containing predicted objects with their category and coordinates.
        groundtruths (list of dict): List of dictionaries containing ground truth objects with their category and coordinates.
        tau (float): Distance threshold for matching predictions with ground truths.

    Returns:
        float: Jaccard metric (tp / (tp + fp + fn)).
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Iterate over each prediction
    for prediction in predictions:
        matched = set()  # Track matched ground truth objects

        # Iterate over each predicted object
        for category, pred_data in prediction.items():
            pred_count = pred_data["num"]
            pred_centers = pred_data["center_cam_3d"]
            pred_dimensions = pred_data["dimensions"]

            # Check if the category exists in ground truths
            if category in groundtruths[0]:
                gt_data = groundtruths[0][category]
                gt_count = gt_data["num"]
                gt_centers = gt_data["center_cam_3d"]
                gt_dimensions = gt_data["dimensions"]
                matched_ground_truths = [False] * gt_count

                # Attempt to match each predicted object with a ground truth
                for j, pred_center in enumerate(pred_centers):
                    best_match_index = -1
                    best_distance = float("inf")

                    # Find the closest unmatched ground truth
                    for i, gt_center in enumerate(gt_centers):
                        if not matched_ground_truths[i]:
                            distance = euclidean(pred_center, gt_center)
                            dimensions_errors = np.mean(
                                np.abs(pred_dimensions[j] - gt_dimensions[i])
                            )
                            if (
                                distance < tau
                                and distance < best_distance
                                and dimensions_errors <= dimensions_tau
                            ):
                                best_distance = distance
                                best_match_index = i

                    # If a match is found, count it as a true positive
                    if best_match_index != -1:
                        true_positives += 1
                        matched_ground_truths[best_match_index] = True
                        matched.add(best_match_index)
                    else:
                        false_positives += 1

                # Add unmatched ground truth objects to false negatives
                false_negatives += gt_count - sum(matched_ground_truths)
            else:
                # All predicted objects of this category are false positives if no matching ground truths
                false_positives += pred_count

        # Account for any ground truths that had no corresponding predictions
        for category, gt_data in groundtruths[0].items():
            if category not in prediction:
                false_negatives += gt_data["num"]

    # Calculate Jaccard metric
    jaccard_metric = true_positives / (
        true_positives + false_positives + false_negatives
    )

    print("true_positives: ", true_positives)
    print("false_positives: ", false_positives)
    print("false_negatives: ", false_negatives)

    return jaccard_metric, true_positives, false_positives, false_negatives


jaccard_metric_sum = 0
total_tp = 0
total_fp = 0
total_fn = 0
total_number_of_scenes = len(final_preprocessed_data_common.keys())

for key in final_preprocessed_data_common.keys():
    jaccard_metric, true_positives, false_positives, false_negatives = compute_jaccard(
        [final_preprocessed_prediction_data_common[key]],
        [final_preprocessed_data_common[key]],
        args.tau,
        args.dimensions_tau,
    )

    total_tp += true_positives
    total_fp += false_positives
    total_fn += false_negatives
    jaccard_metric_sum += jaccard_metric


print(
    "Average Jaccard metric for scene matching: {}".format(
        round(jaccard_metric_sum / total_number_of_scenes, 4)
    )
)

print("Average TP: ", total_tp / total_number_of_scenes)
print("Average FP: ", total_fp / total_number_of_scenes)
print("Average FN: ", total_fn / total_number_of_scenes)
