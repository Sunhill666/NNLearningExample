from enum import Enum


class NeuralNetwork(Enum):
    AlexNet = "AlexNet"
    VGG11 = "VGGNet_11"
    VGG13 = "VGGNet_13"
    VGG16 = "VGGNet_16"
    VGG19 = "VGGNet_19"
    VGGNetA = VGG11
    VGGNetB = VGG13
    VGGNetD = VGG16
    VGGNetE = VGG19


class NetType(Enum):
    Origin = "Origin"
    Simplified = "Simplified"
