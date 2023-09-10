import os.path

import torch

from AlexNet.implement import AlexNetImplement


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    # "origin" and "optimized" are available for model option
    for _model in ["origin", "optimized"]:
        alex_net = AlexNetImplement(_model, root_path, 20, device)
        alex_net.work()


if __name__ == "__main__":
    main()
