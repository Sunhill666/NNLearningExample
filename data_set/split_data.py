import os
from shutil import copy, rmtree
import random


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # If the folder exists, delete the original folder and then recreate it.
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    # Guaranteed random reproducibility
    random.seed(0)

    # Divide 10% of the data in the data set into the validation set, and the remaining 90% into the training set.
    split_rate = 0.1

    # Point to your extracted `flower_photos` folder
    cwd = os.getcwd()
    data_root = os.path.join(cwd, "data_set")
    origin_flower_path = os.path.join(data_root, "flower_photos")
    assert os.path.exists(origin_flower_path), "path '{}' does not exist.".format(origin_flower_path)

    flower_class = [
        cla for cla in os.listdir(origin_flower_path) if os.path.isdir(os.path.join(origin_flower_path, cla))
    ]

    # Create a folder to save the training set, verification set and prediction set.
    train_root = os.path.join(data_root, "train")
    mk_file(train_root)
    for _class in flower_class:
        # Create folders corresponding to each category.
        mk_file(os.path.join(train_root, _class))

    valid_root = os.path.join(data_root, "valid")
    mk_file(valid_root)
    for _class in flower_class:
        mk_file(os.path.join(valid_root, _class))

    predict_root = os.path.join(data_root, "predict")
    mk_file(predict_root)
    for _class in flower_class:
        mk_file(os.path.join(predict_root, _class))

    for _class in flower_class:
        cla_path = os.path.join(origin_flower_path, _class)
        images = os.listdir(cla_path)
        num = len(images)
        # Index of randomly sampled validation set.
        eval_index = random.sample(images, k=int(num*split_rate))
        for index, image in enumerate(images):
            # Copy the files assigned to the validation set to the appropriate directory.
            if image in eval_index:
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(valid_root, _class)
                copy(image_path, new_path)
            else:
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, _class)
                copy(image_path, new_path)
            # Process bar.
            print(f"\r[{_class}] processing [{index + 1} / {num}]", end="")
        print()

    print("Processing done!")


if __name__ == '__main__':
    main()
