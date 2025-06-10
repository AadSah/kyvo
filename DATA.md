# Kyvo Dataset and Codebooks Details

This document provides details about the dataset and codebooks provided in the `kyvo-datasets-and-codebooks` repository. We will provide the details about each of the folders in the repository and the contents of each folder.

## Data Generation Pipeline

The pipeline that we follow to generate the pre-tokenized data is as follows:

* **3D Scenes**: 3D Scene JSON --> Serialized 3D Scene --> Tokenized 3D Scene
* **Images**: Image --> VQGAN Codebook Indices --> Tokenized Image
* **Text**: Text --> Tokenized Text




## Pre-tokenized Data

The `pretokenized-data` folder contains all the pre-tokenized data for the datasets used in the Kyvo project. The pre-tokenized data is stored in the following structure:

```python
pretokenized-data/
|-- clevr/
|   |-- 3d-scenes/ # contains all pre-tokenized 3D scenes for CLEVR for all tasks
|   |-- images/ # contains all pre-tokenized images for CLEVR for all tasks
|   |-- text/ # contains all pre-tokenized text for CLEVR for all tasks
|-- objaworld/
|   |-- 3d-scenes/ # contains all pre-tokenized 3D scenes for ObjaWorld for all tasks
|   |-- images/ # contains all pre-tokenized images for ObjaWorld for all tasks
|-- objectron/
|   |-- 3d-scenes/ # contains all pre-tokenized 3D scenes for Objectron for all tasks
|   |-- images/ # contains all pre-tokenized images for Objectron for all tasks
```


For a given task, an input can be any combination of 3d-scenes, images, and text. The output can be any combination of images, text, and 3d-scenes. In the following table we outline the tasks for each dataset and the corresponding input and output data that are needed for each task.

| **Task**               | **Input Image**    | **Input 3D Scene**     | **Input Text**    | **Output Image**   | **Output 3D Scene**    | **Output Text**   |
|:----------------------:|:------------------:|:----------------------:|:-----------------:|:------------------:|:-----------------------:|:-----------------:|
| **CLEVR**             |                     |                        |                   |                    |                         |                   |
| Rendering             |         𐄂           |           ✓            |        𐄂          |           ✓         |            𐄂            |          𐄂        |
| Recognition           |          ✓          |           𐄂            |         𐄂         |        𐄂            |           ✓             |          𐄂        |
| Instruction-Following |             ✓       |            ✓           |         ✓         |        ✓            |           ✓           |        𐄂        |
| Question-Answering    |             ✓       |            ✓           |         ✓         |        𐄂            |           𐄂           |        ✓        |
|            |                     |                        |                   |                    |                         |                   |
| **ObjaWorld**         |                     |                        |                   |                    |                         |                   |
| Rendering             |         𐄂           |           ✓            |        𐄂          |           ✓         |            𐄂            |          𐄂        |
| Recognition           |          ✓          |           𐄂            |         𐄂         |        𐄂            |           ✓             |          𐄂        |
|            |                     |                        |                   |                    |                         |                   |
| **Objectron**         |                     |                        |                   |                    |                         |                   |
| Recognition           |          ✓          |           𐄂            |         𐄂         |        𐄂            |           ✓             |          𐄂        |

For the exact files that correspond to the input and output data for each task, please refer to the corresponding configuration files in the `configs/llama3_2/train` folder.


## VQGAN Models and Codebooks

The `vqgan-models-and-codebooks` folder contains all the VQGAN model checkpoints and codebooks for the datasets used in the Kyvo project. The VQGAN model checkpoints and codebooks are stored in the following structure:

```python
vqgan-models-and-codebooks/
|-- clevr/
|   |-- 2024-10-10T09-21-36_custom_vqgan_CLEVR-LARGE/ # contains the VQGAN model checkpoint for CLEVR
|   |-- custom_vqgan_embedding_1024CLEVRLARGE_256dim.npy # contains the VQGAN codebook for CLEVR
|-- objaworld/
|   |-- 2025-01-17T09-02-22_custom_vqgan_SYNTHETIC_LIVINGROOM_PARK_LARGE_EP100/ # contains the VQGAN model checkpoint for ObjaWorld
|   |-- custom_vqgan_embedding_256SYNTHETIC_LIVINGROOM_PARK_LARGE_EP100_256dim.npy # contains the VQGAN codebook for ObjaWorld
|-- objectron/
|   |-- 2024-11-03T05-41-42_custom_vqgan_OMNI3D_OBJECTRON_ep200/ # contains the VQGAN model checkpoint for Objectron
|   |-- custom_vqgan_embedding_256Omni3D-OBJECTRON_256dim.npy # contains the VQGAN codebook for Objectron
```

## Images and Scenes for Evaluation

The `images-and-scenes-for-evaluation` folder contains all the groundtruth images and scenes for the datasets used in the Kyvo project. The images and scenes are used to compute the evaluation metrics for the different tasks. The images and scenes are stored in the following structure:

```python
images-and-scenes-for-evaluation/
|-- clevr/ # contains all images and scenes for evaluation for CLEVR
|-- objaworld/ # contains all images and scenes for evaluation for ObjaWorld
|-- objectron/ # contains all scenes for evaluation for Objectron
```



