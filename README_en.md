# Neural Network Learning Example
> [中文介绍](README.md) | English Introduction

## Introduction

This project provides some examples in the process of learning neural networks using `PyTorch`, including some classic neural network models and their simplified versions.

## Dataset Usage

### Follow steps below:

1. Click [Flower dataset](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz) to download dataset and put it in `data_set` directory or

    ```bash
    cd data_set

    # Choose one of the following two methods：

    # Using wget to download
    wget https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

    # Using curl to download
    curl https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz -o flower_photos.tgz
    ```
2. Extract dataset.

    ```bash
    tar -zxvf flower_photos.tgz
    ```

3. Run `split_data.py` to automatically split the dataset into training set `train` and validation set `valid`, and generate the prediction set `predict` empty directory.

    ```bash
    cd ..
    python ./data_set/split_data.py
    ```

4. Finally, the directory structure of `data_set` is as follows:

    ```text
    data_set                                                                                                                                                                                                                                                                                         ─╯
    ├── flower_photos (Extracted dataset folder, 3670 samples)
    ├── predict (Generated prediction set directory, search for pictures on the Internet and put them in the corresponding directory, or copy the pictures under the validation set)
    ├── train (Generated training set, 3306 samples)
    └── valid (Generated validation set, 364 samples)
    ```

## Get Started

1. Clone the repository to local.

    ```bash
    git clone https://github.com/Sunhill666/NNLearningExample.git
    ```

2. Change directory to the repository.

    ```bash
    cd NNLearningExample
    # Create and activate virtual environment (optional)
    conda create -n NNLearningExample python=3.11
    conda activate NNLearningExample
    ```
3. Install [PyTorch](https://pytorch.org/get-started/locally/)。
4. Modify the `main.py` if u need, run it.

    ```bash
    vim main.py
    python main.py
    ```
