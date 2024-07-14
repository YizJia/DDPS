from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnet18_x05, resnet18_x025
from .mobilenetv2 import mobilenet_v2
from .mobilenetv3 import mobilenet_v3_large, mobilenet_v3_small

imagenet_model_dict = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet18_x05": resnet18_x05,
    "resnet18_x025": resnet18_x025,
    "mobilenet_v2": mobilenet_v2,
    "mobilenet_v3_large": mobilenet_v3_large,
    "mobilenet_v3_small": mobilenet_v3_small,
}