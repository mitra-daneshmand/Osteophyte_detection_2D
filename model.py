import pretrainedmodels
import warnings
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from args import parse_args


class FCViewer(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class PretrainedModel(nn.Module):
    def __init__(self, backbone, drop, ncls, pretrained=True):
        super().__init__()
        if pretrained:
            model = pretrainedmodels.__dict__[backbone](num_classes=1000, pretrained='imagenet')
        else:
            model = pretrainedmodels.__dict__[backbone](num_classes=1000, pretrained=None)
        self.encoder = list(model.children())[:-2]

        self.encoder.append(nn.AdaptiveAvgPool2d(1))
        self.encoder = nn.Sequential(*self.encoder)

        if drop > 0:
            self.fc = nn.Sequential(FCViewer(),
                                    nn.Dropout(drop),
                                    nn.Linear(model.last_linear.in_features, ncls))
        else:
            self.fc = nn.Sequential(
                FCViewer(),
                nn.Linear(model.last_linear.in_features, ncls)
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x


class KneeNet(nn.Module):
    def __init__(self, backbone_net, drop, pretrained=True):
        super(KneeNet, self).__init__()
        backbone = PretrainedModel(backbone_net, 1, 1, pretrained)

        self.features = backbone.encoder

        # OST
        self.classifier_ost = nn.Sequential(nn.Dropout(p=drop),
                                            nn.Linear(backbone.fc[-1].in_features, 1))

    def forward(self, x):
        o = self.features(x.float())
        feats = o.view(o.size(0), -1)
        return self.classifier_ost(feats)


VGG_TYPES = {'vgg11': torchvision.models.vgg11,
             'vgg11_bn': torchvision.models.vgg11_bn,
             'vgg13': torchvision.models.vgg13,
             'vgg13_bn': torchvision.models.vgg13_bn,
             'vgg16': torchvision.models.vgg16,
             'vgg16_bn': torchvision.models.vgg16_bn,
             'vgg19_bn': torchvision.models.vgg19_bn,
             'vgg19': torchvision.models.vgg19}


class Custom_VGG(nn.Module):

    def __init__(self,
                 ipt_size=(128, 128),
                 pretrained=True,
                 vgg_type='vgg19_bn',
                 num_classes=1000):
        super(Custom_VGG, self).__init__()
        args = parse_args()
        # load convolutional part of vgg
        # assert vgg_type in VGG_TYPES, "Unknown vgg_type '{}'".format(vgg_type)
        vgg_loader = VGG_TYPES[args.backbone]
        vgg = vgg_loader(pretrained=pretrained)
        self.features = vgg.features


        # init fully connected part of vgg
        test_ipt = Variable(torch.zeros(1,3,ipt_size[0],ipt_size[1]))
        test_out = vgg.features(test_ipt)
        self.n_features = test_out.size(1) * test_out.size(2) * test_out.size(3)
        self.classifier = nn.Sequential(nn.Linear(self.n_features, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(),
                                        nn.Linear(4096, args.n_classes)
                                       )
        self._init_classifier_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _init_classifier_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


############################### resnet10 ###############################


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock2D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock2D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet2D(nn.Module):
    def __init__(self, block, layers, num_classes=1000, in_channels=24,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 ngf=64, norm_layer=None, last_sigmoid=False):
        super(ResNet2D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.last_sigmoid = last_sigmoid
        if self.last_sigmoid:
            warnings.warn("モデルの出力にsigmoid関数を施します")
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, ngf, layers[0])
        self.layer2 = self._make_layer(block, ngf * 2, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, ngf * 4, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, ngf * 8, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(ngf * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.forward_until_avgpool(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if self.last_sigmoid:
            x = F.sigmoid(x)

        return x

    def forward_until_avgpool(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet10_2D(num_classes, in_channels, ngf, last_sigmoid=False):
    """Constructs a ResNet-10 model.
    """
    model = ResNet2D(BasicBlock2D, [1, 1, 1, 1], num_classes, in_channels, ngf=ngf, last_sigmoid=last_sigmoid)
    return model


