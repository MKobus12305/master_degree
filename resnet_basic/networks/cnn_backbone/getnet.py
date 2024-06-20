from .resnet import *

def get_backnet(backbone_name, pretrained):
    channel_settings = [2048, 1024, 512, 256]
    backbone = resnet50(pretrained=pretrained)

    return channel_settings, backbone