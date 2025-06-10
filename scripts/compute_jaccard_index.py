import json
import numpy as np
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Compute Jaccard Index")

parser.add_argument(
    "--generated_folder",
    required=True,
    type=str,
    help="Path to the folder containing the predicted scenes",
)

parser.add_argument(
    "--groundtruth_folder",
    required=True,
    type=str,
    help="Path to the folder containing the ground truth scenes",
)

parser.add_argument(
    "--tau",
    required=True,
    type=float,
    default=0.25,
    help="Distance threshold for matching objects",
)

parser.add_argument(
    "--dataset",
    required=True,
    type=str,
    help="Dataset used for training: choose from [clevr, objaworld]",
)

args = parser.parse_args()


# Function to compute Euclidean distance between two 2D coordinates
def dist(coords1, coords2):
    return np.linalg.norm(np.array(coords1) - np.array(coords2))


# Function to compute the Jaccard Index
def compute_jaccard_index(ground_truth_scenes, predicted_scenes, tau):

    total_number_of_scenes = len(ground_truth_scenes)
    assert total_number_of_scenes == len(predicted_scenes)
    total_jaccard_index = 0

    for gt_scene, pred_scene in zip(ground_truth_scenes, predicted_scenes):
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        if args.dataset == "clevr":
            assert (
                gt_scene["image_filename"] == pred_scene["key"]
                or gt_scene["image_filename"].split(".png")[0]
                == pred_scene["key"].split(".png")[0]
            ), f"Ground truth and predicted scene mismatch: {gt_scene['image_filename']} vs {pred_scene['key']}"
        elif args.dataset == "objaworld":
            assert (
                gt_scene["image_filename"].split(".png")[0]
                == pred_scene["key"].split(".png_")[0]
            ), f"Ground truth and predicted scene mismatch: {gt_scene['image_filename']} vs {pred_scene['key']}"

        gt_objects = gt_scene["objects"]
        pred_objects = pred_scene["objects"]

        # Create flags to track matched ground truth objects
        gt_matched = [False] * len(gt_objects)

        # Match predictions to ground truth
        for pred_obj in pred_objects:
            matched = False
            for i, gt_obj in enumerate(gt_objects):
                if not gt_matched[i]:  # Check if ground truth object is unmatched
                    # Check attribute equality and distance condition
                    try:
                        if args.dataset == "clevr":
                            if (
                                pred_obj["size"] == gt_obj["size"]
                                and pred_obj["color"] == gt_obj["color"]
                                and pred_obj["material"] == gt_obj["material"]
                                and pred_obj["shape"] == gt_obj["shape"]
                                and dist(
                                    pred_obj["3d_coords"][:2], gt_obj["3d_coords"][:2]
                                )
                                < tau
                            ):
                                gt_matched[i] = True  # Mark ground truth as matched
                                matched = True
                                true_positives += 1
                                break
                        elif args.dataset == "objaworld":
                            if (
                                pred_obj["shape"] == gt_obj["shape"].split("-")[0]
                                and dist(
                                    pred_obj["3d_coords"][:2], gt_obj["3d_coords"][:2]
                                )
                                < tau
                                and abs(
                                    pred_obj["3d_coords"][2] - gt_obj["3d_coords"][2]
                                )
                                < 0.05
                                and abs(pred_obj["3d_coords"][3] - gt_obj["rotation"])
                                < 0.15
                            ):
                                gt_matched[i] = True  # Mark ground truth as matched
                                matched = True
                                true_positives += 1
                                break
                    except:
                        continue  # Skip if any attribute is missing
            if not matched:
                false_positives += 1

        # Count unmatched ground truths
        false_negatives += gt_matched.count(False)

        if (true_positives + false_positives + false_negatives) == 0:
            jaccard_index = -1
        else:
            jaccard_index = true_positives / (
                true_positives + false_positives + false_negatives
            )
        total_jaccard_index += jaccard_index

    # Compute Jaccard Index

    jaccard_index = total_jaccard_index / total_number_of_scenes
    return jaccard_index


# find the predicted scenes json file
folder = os.path.basename(args.generated_folder)
print(f"Computing Jaccard Index for {folder}")
with open(f"{args.generated_folder}/predicted_scenes_{folder}.json", "r") as f:
    predicted_data = json.load(f)

ground_truth_data = {"scenes": []}

if args.dataset == "clevr":
    all_gt_jsons = os.listdir(args.groundtruth_folder)
    all_gt_jsons = sorted(all_gt_jsons)
    for gt_json in all_gt_jsons:
        with open(f"{args.groundtruth_folder}/{gt_json}", "r") as f:
            ground_truth_data["scenes"].append(json.load(f))
elif args.dataset == "objaworld":
    for pred_scene in predicted_data["scenes"]:
        with open(
            "{}/output-{}-large-10K-6/scenes/{}.json".format(
                args.groundtruth_folder,
                pred_scene["key"].split("png_")[1],
                pred_scene["key"].split(".png_")[0],
            ),
            "r",
        ) as f:
            ground_truth_data["scenes"].append(json.load(f))

ground_truth_scenes = ground_truth_data["scenes"]
predicted_scenes = predicted_data["scenes"]

ground_truth_scenes = ground_truth_scenes[: len(predicted_scenes)]

jaccard_index = compute_jaccard_index(
    ground_truth_scenes, predicted_scenes, tau=args.tau
)

print(f"Jaccard Index: {jaccard_index:.4f}")

# print the Jaccard Index to a file in the same folder
with open(f"{args.generated_folder}/jaccard_index_tau-{args.tau}.txt", "w") as f:
    f.write(f"Jaccard Index: {jaccard_index:.4f}")
