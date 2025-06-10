import os
import numpy as np
from skimage.io import imread
from skimage.metrics import structural_similarity as ssim
import glob
from tqdm import tqdm
import re
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description="Compute L2 Loss and SSIM")

parser.add_argument(
    "--generated_folder",
    required=True,
    type=str,
    help="Path to the folder containing the generated images",
)

parser.add_argument(
    "--groundtruth_folder",
    required=True,
    type=str,
    help="Path to the folder containing the ground truth images",
)

parser.add_argument(
    "--output_folder",
    required=True,
    type=str,
    help="Path to the folder where the results will be saved",
)


args = parser.parse_args()


def compute_l2_loss(img1, img2):
    return np.mean((img1 - img2) ** 2)


groundtruth_folder = args.groundtruth_folder
generated_folder = args.generated_folder

# Initialize accumulators for L2 loss and SSIM
total_l2_loss = 0
total_ssim = 0
image_count = 0

# Get sorted list of image files
groundtruth_files = sorted(glob.glob(os.path.join(groundtruth_folder, "*.png")))
generated_files = sorted(glob.glob(os.path.join(generated_folder, "*.png")))

groundtruth_files = groundtruth_files[: len(generated_files)]


if len(generated_files) == 0:
    raise ValueError("No images found in GENERATED folder.")


# Iterate through the image pairs
print("Computing L2 Loss and SSIM for image pairs...")
for gt_file, gen_file in tqdm(
    zip(groundtruth_files, generated_files), total=len(groundtruth_files)
):
    # Assert filenames match up to the folder and prefix differences
    gt_filename = os.path.basename(gt_file)
    gen_filename = os.path.basename(gen_file)

    assert (
        gt_filename == gen_filename.split("generated_")[1]
    ), f"File names do not match: {gt_filename} and {gen_filename}"

    # Load images
    groundtruth_img = imread(gt_file, as_gray=True)  # Load as grayscale for SSIM
    generated_img = imread(gen_file, as_gray=True)

    # resize groundtruth image to generated image size
    groundtruth_img = np.array(
        Image.fromarray(groundtruth_img).resize(
            (generated_img.shape[0], generated_img.shape[1])
        )
    )

    # Ensure the images have the same shape
    if groundtruth_img.shape != generated_img.shape:
        raise ValueError(f"Image shapes do not match: {gt_file} and {gen_file}")

    # Compute L2 loss and SSIM
    l2_loss = compute_l2_loss(groundtruth_img, generated_img)
    image_ssim = ssim(
        groundtruth_img,
        generated_img,
        data_range=generated_img.max() - generated_img.min(),
    )

    # Accumulate results
    total_l2_loss += l2_loss
    total_ssim += image_ssim
    image_count += 1

# Compute averages
average_l2_loss = total_l2_loss / image_count
average_ssim = total_ssim / image_count

# Print results
print("\nComputation Complete!")
print(f"Average L2 Loss: {average_l2_loss}")
print(f"Average SSIM: {average_ssim}")

# Write results to a file in the same folder
with open(f"{args.output_folder}/ssim_l2.txt", "w") as f:
    f.write(f"Average L2 Loss, Average SSIM:\n")
    f.write(f"{average_l2_loss}, {average_ssim}\n")
    f.write(f"Total L2 Loss, Total SSIM:\n")
    f.write(f"{total_l2_loss}, {total_ssim}\n")
    f.write(f"Image Count: {image_count}\n")
