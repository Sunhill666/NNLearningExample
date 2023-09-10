import os
import sys

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from AlexNet.model import OriginAlexNet, SimplifiedAlexNet


class AlexNetImplement:
    def __init__(self, model, root_path, epochs=10, device="cpu"):
        train_path = os.path.join(root_path, "data_set", "train")
        valid_path = os.path.join(root_path, "data_set", "valid")
        predict_path = os.path.join(root_path, "data_set", "predict")
        assert os.path.exists(train_path), f"folder: '{train_path}' dose not exist."
        assert os.path.exists(valid_path), f"folder: '{valid_path}' dose not exist."
        assert os.path.exists(predict_path), f"folder: '{predict_path}' dose not exist."
        if not os.path.exists(os.path.join(root_path, "model")):
            os.mkdir(os.path.join(root_path, "model"))

        self._root_path = root_path
        self._data_path = os.path.join(root_path, "data_set")
        self._epochs = epochs
        self._device = device
        if model == "origin":
            self._model = OriginAlexNet(num_classes=len(self._class_dict)).to(self._device)
            self._save_path = os.path.join(root_path, "model", "OriginAlexNet.pth")
        else:
            self._model = SimplifiedAlexNet(num_classes=len(self._class_dict)).to(self._device)
            self._save_path = os.path.join(root_path, "model", "SimplifiedAlexNet.pth")

    @property
    def _class_dict(self):
        train_path = os.path.join(self._root_path, "data_set", "train")
        class_list = [_dir for _dir in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, _dir))]
        # {0: 'roses', 1: 'sunflowers', 2: 'daisy', 3: 'dandelion', 4: 'tulips'}
        return dict((index, class_name) for index, class_name in enumerate(class_list))

    def work(self):
        self.train()
        self.predict()

    def train(self):
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        valid_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        print(f"Using {self._device} device to train model {self._model.__class__.__name__}")

        train_set = datasets.ImageFolder(os.path.join(self._data_path, "train"), transform=train_transform)
        train_num = len(train_set)
        valid_set = datasets.ImageFolder(os.path.join(self._data_path, "valid"), transform=valid_transform)
        valid_num = len(valid_set)

        batch_size = 32
        worker_num = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])
        print(f"Using {worker_num} workers to load data")

        # If `multiprocessing_context` keeps default, the DataLoader will slow down on macOS(probability Apple Silicon).
        # So we need to set `multiprocessing_context` to "fork" to solve this problem.
        # And I advise that set `persistent_workers` to True to speed up DataLoader on training.
        # See more on https://discuss.pytorch.org/t/data-loader-multiprocessing-slow-on-macos/131204
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size, True,
            multiprocessing_context="fork", num_workers=worker_num, persistent_workers=True
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_set, batch_size, True,
            multiprocessing_context="fork", num_workers=worker_num
        )
        print(f"Using {train_num} images for training, {valid_num} images for validation")

        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-4)

        best_acc = 0.0
        for epoch in range(self._epochs):
            # Training
            self._model.train()
            running_loss = 0.0
            train_bar = tqdm(train_loader, file=sys.stdout)
            for step, data in enumerate(train_bar):
                images, labels = data
                # Initialize the model’s parameter gradients to 0.
                optimizer.zero_grad()
                # Forward propagation calculates predicted values.
                outputs = self._model(images.to(self._device))
                # Calculate current loss.
                loss = loss_function(outputs, labels.to(self._device))
                # Back propagation computes gradients.
                loss.backward()
                # Update all parameters.
                optimizer.step()
                running_loss += loss.item()
                train_bar.desc = f"[{epoch + 1}/{self._epochs}] Loss: {loss:.3f}"

            # Validation
            self._model.eval()
            acc = 0.0
            with torch.no_grad():
                valid_bar = tqdm(valid_loader, file=sys.stdout)
                for data in valid_bar:
                    images, labels = data
                    outputs = self._model(images.to(self._device))
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, labels.to(self._device)).sum().item()

            val_acc = acc / valid_num
            print(f"[{epoch + 1}/{self._epochs}] Loss: {running_loss / len(train_loader):.3f}, Accuracy: {val_acc:.3f}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self._model.state_dict(), self._save_path)

        print(f"Finished Training. Best Accuracy: {best_acc:.3f}")

    def predict(self):
        if not os.path.exists(self._save_path):
            print("Model dose not exist, train now.")
            self.train()
        else:
            self._model.load_state_dict(torch.load(self._save_path))

        data_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        for value in self._class_dict.values():
            print(f"********** Predict {value} class now")
            predict_class_path = os.path.join(self._data_path, "predict", value)
            image_list = [
                os.path.join(predict_class_path, file) for file in os.listdir(predict_class_path) if
                os.path.isfile(os.path.join(predict_class_path, file)) and not file.startswith('.')
            ]
            for image_path in image_list:
                image = torch.unsqueeze(data_transform(Image.open(image_path)), dim=0)

                self._model.eval()
                with torch.no_grad():
                    output = torch.squeeze(self._model(image.to(self._device))).cpu()
                    predict = torch.softmax(output, dim=0)
                    predict_class = torch.argmax(predict).numpy()
                    predict_class = int(predict_class)

                print("Result:")
                print(f"Class: {self._class_dict[predict_class]:10} Probability: {predict[predict_class].numpy():.3}")
