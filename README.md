# Neural Network Learning Example
> 中文介绍 | [English Introduction](README_en.md)

## 介绍

此项目为在使用 `PyTorch` 学习神经网络过程中的一些例子，包括一些经典的神经网络模型及其简化版本。

## 数据集使用说明

### 使用步骤如下：

1. 点击链接下载 [花卉数据集](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz) 放到 `data_set` 目录下或

    ```bash
    cd data_set

    # 以下两种方式任选其一：

    # 使用wget下载
    wget https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

    # 使用curl下载
    curl https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz -o flower_photos.tgz
    ```
2. 解压数据集。

    ```bash
    tar -zxvf flower_photos.tgz
    ```

3. 执行 `split_data.py` 脚本自动将数据集划分成训练集 `train` 和验证集 `valid` ，并生成预测集 `predict` 空目录。

    ```bash
    cd ..
    python ./data_set/split_data.py
    ```

4. 最终 `data_set` 目录结构如下：
    ```text
    data_set                                                                                                                                                                                                                                                                                         ─╯
    ├── flower_photos（解压的数据集文件夹，3670个样本）
    ├── predict（生成的预测集目录，自行在网络上搜寻图片放到对应目录下，或复制验证集下的图片）
    ├── train（生成的训练集，3306个样本）
    └── valid（生成的验证集，364个样本）
    ```

## Get Started

1. 克隆项目到本地。

    ```bash
    git clone https://github.com/Sunhill666/NNLearningExample.git
    ```

2. 移动到项目目录下。

    ```bash
    cd NNLearningExample
    # 创建并激活虚拟环境（可选项）
    conda create -n NNLearningExample python=3.11
    conda activate NNLearningExample
    ```
3. 安装 [PyTorch](https://pytorch.org/get-started/locally/)。
4. 按需要修改 `main.py` 并运行。

    ```bash
    vim main.py
    python main.py
    ```
