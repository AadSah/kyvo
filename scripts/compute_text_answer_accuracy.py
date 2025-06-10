import json
import re
from tqdm import tqdm
import argparse

# Parser for command line arguments
parser = argparse.ArgumentParser(description="Compute Text Answer Accuracy")

parser.add_argument(
    "--groundtruth_file",
    required=True,
    type=str,
    help="Path to the ground truth answers JSON file.",
)

parser.add_argument(
    "--predicted_file",
    required=True,
    type=str,
    help="Path to the predicted answers JSON file.",
)

args = parser.parse_args()


def clean_answer(answer):
    """
    Clean the answer by removing special tokens like [TEXT-START], [TEXT-END], and <|end_of_text|>.
    """
    return re.sub(r"\[TEXT-START\]|\[TEXT-END\]|<\|end_of_text\|>", "", answer).strip()


def evaluate_accuracy(groundtruth_file, predicted_file):
    """
    Evaluate the accuracy of predicted answers against ground truth answers.

    Args:
    - groundtruth_file (str): Path to the ground truth answers JSON file.
    - predicted_file (str): Path to the predicted answers JSON file.

    Returns:
    - float: Accuracy of the model (in percentage).
    """
    # Load ground truth and predicted answers
    with open(groundtruth_file, "r") as gt_file:
        groundtruth_data = json.load(gt_file)
    with open(predicted_file, "r") as pred_file:
        predicted_data = json.load(pred_file)

    # Assertions to ensure data integrity
    assert "answers" in groundtruth_data, "Ground truth JSON missing 'answers' key."
    assert "answers" in predicted_data, "Predicted answers JSON missing 'answers' key."
    assert len(groundtruth_data["answers"]) == len(
        predicted_data["answers"]
    ), "Mismatch in number of ground truth and predicted answers."

    groundtruth_answers = groundtruth_data["answers"]
    predicted_answers = predicted_data["answers"]

    correct_count = 0

    # Progress bar for evaluation
    for gt, pred in tqdm(
        zip(groundtruth_answers, predicted_answers),
        total=len(groundtruth_answers),
        desc="Evaluating",
    ):
        # Assertions for data consistency
        assert (
            gt["image_filename"].split("_scene_")[1]
            == pred["image_filename"].split("_scene_")[1]
        ), f"Image filename mismatch: {gt['image_filename']} vs {pred['image_filename']}"

        # Clean answers to extract meaningful text
        gt_answer = clean_answer(gt["answer"])
        pred_answer = clean_answer(pred["answer"])

        # Compare and count correct predictions
        if gt_answer == pred_answer:
            correct_count += 1

    # Calculate accuracy
    accuracy = (correct_count / len(groundtruth_answers)) * 100
    return accuracy


# Evaluate and print accuracy
accuracy = evaluate_accuracy(args.groundtruth_file, args.predicted_file)
print(f"Model Text Answer Accuracy: {accuracy:.2f}%")
