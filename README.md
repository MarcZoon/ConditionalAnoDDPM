# Conditional AnoDDPM



This is the codebase for my thesis, Conditional AnoDDPM.

The code is a fork [Improved Diffusion](https://github.com/openai/improved-diffusion), and implements Simplex noise as an alternative to Gaussian noise.

# Usage

This section of the README walks through how to train and sample from a model.

## Installation

Clone this repository and navigate to it in your terminal. Then run:

```
pip install -e .
```

This should install the `canoddpm` python package that the scripts depend on.

## Preparing Data

You can download the BraTS2020 dataset from [here](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation).

Run the `datasets/brats.py` script to pre-process the dataset.

## Training

The `train_example.sh` is an example script to train a model. It uses the `scripts/multi_mri_train.py` script.

## Evaluation

Use the `scripts/multi_mri_eval.py` script to evaluate a model.
