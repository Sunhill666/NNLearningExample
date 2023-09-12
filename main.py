import os.path

import torch

from implement import NNImplement
from utils import NetType, NeuralNetwork


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    '''
    Each model best accuracy in 10 epochs:
    OriginAlexNet: 0.832
    '''
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    alex_net = NNImplement(NeuralNetwork.VGG16, NetType.Origin, root_path, 100, device)
    alex_net.work()


if __name__ == "__main__":
    main()
