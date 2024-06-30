import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBlock(nn.Module):
    def __init__(self, f1, f2, use_groupnorm=True, groups=8, dilation=1, transpose=False):
        super().__init__()
        self.transpose = transpose
        self.conv = nn.Conv2d(f1, f2, (3, 3), dilation=dilation, padding=dilation)
        if self.transpose:
            self.convt = nn.ConvTranspose2d(
                f1, f1, (3, 3), dilation=dilation, stride=2, padding=dilation, output_padding=1
            )
        if use_groupnorm:
            self.bn = nn.GroupNorm(groups, f1)
        else:
            self.bn = nn.GroupNorm(8, f1)

    def forward(self, x):
        # x = F.dropout(x, 0.04, self.training)
        x = self.bn(x)
        if self.transpose:
            # x = F.upsample(x, scale_factor=2, mode='bilinear')
            x = F.relu(self.convt(x))
            # x = x[:, :, :-1, :-1]
        x = F.relu(self.conv(x))
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(8, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.GroupNorm(8, planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(8, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetOriginal(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNetOriginal, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.GroupNorm(8, 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 196, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8, planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNetDepth(nn.Module):
    def __init__(self, out_dim=1, use_torch_version=True, freeze_backbone=False, decoder_layers=10, pretrained=False):
        super().__init__()
        if use_torch_version:
            print(f'load torch version, imagenet pretrained={pretrained==True}')
            resnet = models.resnet50(pretrained=pretrained == True)
        else:
            # this one uses GroupNorm, buggy
            resnet = ResNetOriginal(Bottleneck, [3, 4, 6, 3])
        del resnet.fc
        if pretrained == 'dcl':
            print(f'load DenseCL: pretrained_weights/densecl_r50_imagenet_200ep.pth')
            s = torch.load('pretrained_weights/densecl_r50_imagenet_200ep.pth', map_location='cpu')
            resnet.load_state_dict(s['state_dict'], strict=False)
        elif pretrained == 'pixp':
            print(f'load PixelPro: pretrained_weights/pixpro_base_r50_100ep_md5_91059202.pth')
            s = torch.load('pretrained_weights/pixpro_base_r50_100ep_md5_91059202.pth', map_location='cpu')
            resnet.load_state_dict(
                {k[len('module.encoder.'):]: v for k, v in s['model'].items() if k.startswith('module.encoder.')}, strict=False)
        elif not isinstance(pretrained, bool):
            raise ValueError(f'unsupported pretrained={pretrained}')

        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            resnet.eval()
            for p in resnet.parameters():
                p.requires_grad = False

        self.final_conv = nn.Conv2d(2048, 8, (3, 3), padding=1)
        if decoder_layers == 3:
            self.decoder = nn.Sequential(
                ConvBlock(8, 128, transpose=True),
                ConvBlock(128, 128, transpose=True),
                ConvBlock(128, out_dim, transpose=True),
            )
        elif decoder_layers == 4:
            self.decoder = nn.Sequential(
                ConvBlock(8, 128, transpose=True),
                ConvBlock(128, 128, transpose=True),
                ConvBlock(128, 128, transpose=True),
                ConvBlock(128, out_dim, transpose=True),
            )
        elif decoder_layers == 5:
            self.decoder = nn.Sequential(
                ConvBlock(8, 128),
                ConvBlock(128, 128, transpose=True),
                ConvBlock(128, 128, transpose=True),
                ConvBlock(128, 128, transpose=True),
                ConvBlock(128, out_dim, transpose=True),
            )
        elif decoder_layers == 6:
            self.decoder = nn.Sequential(
                ConvBlock(8, 128),
                ConvBlock(128, 128),
                ConvBlock(128, 128, transpose=True),
                ConvBlock(128, 128, transpose=True),
                ConvBlock(128, 128, transpose=True),
                ConvBlock(128, out_dim, transpose=True),
            )
        elif decoder_layers == 7:
            self.decoder = nn.Sequential(
                ConvBlock(8, 128),
                ConvBlock(128, 128),
                ConvBlock(128, 128),
                ConvBlock(128, 128, transpose=True),
                ConvBlock(128, 128, transpose=True),
                ConvBlock(128, 128, transpose=True),
                ConvBlock(128, out_dim, transpose=True),
            )
        elif decoder_layers == 10:
            self.decoder = nn.Sequential(
                ConvBlock(8, 128),
                ConvBlock(128, 128),
                ConvBlock(128, 128),
                ConvBlock(128, 128),
                ConvBlock(128, 128),
                ConvBlock(128, 128, transpose=True),
                ConvBlock(128, 128, transpose=True),
                ConvBlock(128, 128, transpose=True),
                ConvBlock(128, 128, transpose=True),
                ConvBlock(128, out_dim, transpose=True),
            )
        # remove last conv stage (3 BottleNecks), feature map size: 1024x16x16 (input image 3x256x256)
        # self.encoder = nn.Sequential(*list(self.resnet._modules.values())[:-2])
        # remove last avgpool, feature map size: 2048x8x8 (input image 3x256x256)
        self.encoder = nn.Sequential(*list(resnet._modules.values())[:-1])

    def train(self, mode=None):
        # print('switch to train')
        if not self.freeze_backbone:
            self.encoder.train()
        self.final_conv.train()
        self.decoder.train()

    def eval(self):
        # print('switch to eval')
        if not self.freeze_backbone:
            self.encoder.eval()
        self.final_conv.eval()
        self.decoder.eval()

    def forward(self, x):
        # print(x.shape)  # [24,3,256,256]
        if self.freeze_backbone:
            with torch.no_grad():
                x = self.encoder(x)
        else:
            x = self.encoder(x)
            # for layer in list(self.resnet._modules.values())[:-2]:
            #     x = layer(x)
        # print(x.shape)  # [24,2048,8,8] if keep all resnet stages
        x = self.final_conv(x)
        x = self.decoder(x)

        return x

    
    
## below from https://github.com/EPFL-VILAB/XTConsistency/blob/1c7a1875376fa60277bcf1baebd1005501f53c99/modules/depth_nets.py

# class ConvBlock(nn.Module):
#     def __init__(self, f1, f2, use_groupnorm=True, groups=8, dilation=1, transpose=False):
#         super().__init__()
#         self.transpose = transpose
#         self.conv = nn.Conv2d(f1, f2, (3, 3), dilation=dilation, padding=dilation)
#         if self.transpose:
#             self.convt = nn.ConvTranspose2d(
#                 f1, f1, (3, 3), dilation=dilation, stride=2, padding=dilation, output_padding=1
#             )
#         if use_groupnorm:
#             self.bn = nn.GroupNorm(groups, f1)
#         else:
#             self.bn = nn.GroupNorm(8, f1)

#     def forward(self, x):
#         # x = F.dropout(x, 0.04, self.training)
#         x = self.bn(x)
#         if self.transpose:
#             # x = F.upsample(x, scale_factor=2, mode='bilinear')
#             x = F.relu(self.convt(x))
#             # x = x[:, :, :-1, :-1]
#         x = F.relu(self.conv(x))
#         return x

# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.GroupNorm(8, planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.GroupNorm(8, planes)
#         self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = nn.GroupNorm(8, planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out

# class ResNetOriginal(nn.Module):

#     def __init__(self, block, layers, num_classes=1000):
#         self.inplanes = 64
#         super(ResNetOriginal, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = nn.GroupNorm(8, 64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 196, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AvgPool2d(7, stride=1)
#         self.fc = nn.Linear(512 * block.expansion, num_classes)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.GroupNorm(8, planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)

#         return x

# class ResNetDepth(TrainableModel):
#     def __init__(self):
#         super().__init__()
#         # self.resnet = models.resnet50()
#         self.resnet = ResNetOriginal(Bottleneck, [3, 4, 6, 3])
#         self.final_conv = nn.Conv2d(2048, 8, (3, 3), padding=1)

#         self.decoder = nn.Sequential(
#             ConvBlock(8, 128),
#             ConvBlock(128, 128),
#             ConvBlock(128, 128),
#             ConvBlock(128, 128),
#             ConvBlock(128, 128),
#             ConvBlock(128, 128, transpose=True),
#             ConvBlock(128, 128, transpose=True),
#             ConvBlock(128, 128, transpose=True),
#             ConvBlock(128, 128, transpose=True),
#             ConvBlock(128, 1, transpose=True),
#         )

#     def forward(self, x):

#         for layer in list(self.resnet._modules.values())[:-2]:
#             x = layer(x)

#         x = self.final_conv(x)
#         x = self.decoder(x)

#         return x

#     def loss(self, pred, target):
#         loss = torch.tensor(0.0, device=pred.device)
#         return loss, (loss.detach(),)
