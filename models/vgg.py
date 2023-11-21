"""
Standard VGG-16
"""

import torch
import torch.nn as nn
import torchvision


class VGG16(nn.Module):
    def __init__(self, pretrained=False, dataset='flickr25k', backbone_layer_train=0):
        super(VGG16, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=pretrained)
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:6])
        self.dataset = dataset
        self.backbone_layer_train = backbone_layer_train

        if self.dataset == 'cifar10':
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7)) # to be compatible with 32x32 input image size
            assert self.backbone_layer_train == 'ALL'
        else:
            self.avgpool = None
            for i, param in enumerate(self.vgg.parameters()):
                if self.backbone_layer_train == 0:
                    param.requires_grad = False
                    continue
                if (i + backbone_layer_train*2) < 30:
                    param.requires_grad = False
 
    def forward(self, x):
        x = self.vgg.features(x)
        if self.avgpool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.vgg.classifier(x)
        return x


def vgg16(pretrained=True, dataset='flickr25k', backbone_layer_train=0, **kwargs):
    model = VGG16(pretrained, dataset, backbone_layer_train)
    return model

