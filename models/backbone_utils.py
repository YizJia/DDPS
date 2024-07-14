from collections import OrderedDict
import torch.nn.functional as F
from torchvision.ops import misc as misc_nn_ops
from torch import nn
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.models._utils import IntermediateLayerGetter

from models.backbone import imagenet_model_dict

class Backbone(nn.Sequential):
    def __init__(self, resnet, model_cfg):
        super(Backbone, self).__init__(
            OrderedDict(
                [
                    ["conv1", resnet.conv1],
                    ["bn1", resnet.bn1],
                    ["relu", resnet.relu],
                    ["maxpool", resnet.maxpool],
                    ["layer1", resnet.layer1],  # res2
                    ["layer2", resnet.layer2],  # res3
                    ["layer3", resnet.layer3],  # res4
                ]
            )
        )
        self.out_channels = model_cfg.BACKBONE_OUTCHANNEL

    def forward(self, x):
        # using the forward method from nn.Sequential
        feat = super(Backbone, self).forward(x)
        return OrderedDict([["feat_res4", feat]])


class Res5Head(nn.Sequential):
    def __init__(self, resnet, model_cfg):
        super(Res5Head, self).__init__(OrderedDict([["layer4", resnet.layer4]]))  # res5
        self.out_channels = model_cfg.RES5HEAD_OUTCHANNEL

    def forward(self, x):
        feat = super(Res5Head, self).forward(x)
        x = F.adaptive_max_pool2d(x, 1)
        feat = F.adaptive_max_pool2d(feat, 1)
        return OrderedDict([["feat_res4", x], ["feat_res5", feat]])


class BackboneWithFPN(nn.Module):
    def __init__(self, backbone, return_layers, in_channels_list, out_channels, extra_blocks=None):
        super(BackboneWithFPN, self).__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)

        return x
        # x_name = [k for k in x.keys()]
        # assert set(self.out_layers).issubset(set(x_name)), "out layers should be a subset of the fpn output's name"

        # if len(self.out_layers) == len(x):
        #     return x
        # else:
        #     out = OrderedDict()
        #     for v in self.out_layers:
        #         out[v] = x[v]
        #     return out

def build_resnet(model_cfg):
    ## Return a model pre-trained on ImageNet
    # resnet = torchvision.models.resnet.__dict__[model_cfg.BACKBONE_NAME](pretrained=pretrained)
    resnet = imagenet_model_dict[model_cfg.BACKBONE_NAME](pretrained=model_cfg.BACKBONE_PRETRAIN)

    # freeze layers
    # dont freeze any layers if the pre-trained model or backbone is not used
    if model_cfg.FREEZE:
        resnet.conv1.weight.requires_grad_(False)
        resnet.bn1.weight.requires_grad_(False)
        resnet.bn1.bias.requires_grad_(False)

    return Backbone(resnet, model_cfg), Res5Head(resnet, model_cfg)

def build_resnet_fpn(model_cfg):
    if model_cfg.FROZENBN2d:
        backbone = imagenet_model_dict[model_cfg.BACKBONE_NAME](
            pretrained=model_cfg.BACKBONE_PRETRAIN,
            # new! same as configs in torchvision
            norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    else:
        backbone = imagenet_model_dict[model_cfg.BACKBONE_NAME](
            pretrained=model_cfg.BACKBONE_PRETRAIN)

    # freeze layers
    # tranable layer number=4, starting from final block
    # under codes belongs to 'FREEZE' parts are different from torchvision
    # dont freeze any layers if the pre-trained model or backbone is not used
    if model_cfg.FREEZE:
        backbone.conv1.weight.requires_grad_(False)
        backbone.bn1.weight.requires_grad_(False)
        backbone.bn1.bias.requires_grad_(False)

    extra_blocks = LastLevelMaxPool()

    assert model_cfg.BACKBONE_NECK, "W or W/O FPN is not clear"

    returned_layers = [1, 2, 3, 4]
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i-1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)

def build_mobilenet_fpn(model_cfg):
    backbone = imagenet_model_dict[model_cfg.BACKBONE_NAME](
        pretrained=model_cfg.BACKBONE_PRETRAIN).features

    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    # stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    stage_indices = [i for i, b in enumerate(backbone)]
    num_stages = len(stage_indices)

    # trainable_layers = model_cfg.TRAINABLE_LABYER
    # # find the index of the layer from which we wont freeze
    # assert 0 <= trainable_layers <= num_stages
    # freeze_before = len(backbone) if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

    # for b in backbone[:freeze_before]:
    #     for parameter in b.parameters():
    #         parameter.requires_grad_(False)

    out_channels = 256
    if model_cfg.BACKBONE_NECK:
        extra_blocks = LastLevelMaxPool()

        # returned_layers = [num_stages - 2, num_stages - 1]
        returned_layers = [3, 6, 13, 17]
        assert min(returned_layers) >= 0 and max(returned_layers) < num_stages
        return_layers = {f'{stage_indices[k]}': str(v) for v, k in enumerate(returned_layers)}

        in_channels_list = [backbone[stage_indices[i]].out_channels for i in returned_layers]
        # different from torchvision
        return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)
    else:
        m = nn.Sequential(
            backbone,
            # depthwise linear combination of channels to reduce their size
            nn.Conv2d(backbone[-1].out_channels, out_channels, 1),
        )
        m.out_channels = out_channels
        return m 