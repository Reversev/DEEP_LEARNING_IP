import torch
from torch import nn
from torchvision.models import *
import numpy as np

# vgg19的features网络通过5个MaxPool将图像尺寸缩小了32倍
# 图像尺寸缩小后分别在：MaxPool2d-5(缩小2倍) ,MaxPool2d-10 （缩小4倍）,MaxPool2d-19（缩小8倍）,
# MaxPool2d-28（缩小16倍）,MaxPool2d-37（缩小32倍）


# define FCN8s network
class FCN8s_VGG(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # num_classes:训练数据的类别
        self.num_classes = num_classes
        model = vgg19(pretrained=True)
        self.base_model = model.features

        # 定义几个需要的层操作，并且使用转置卷积将特征映射进行升维
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2,
                                          padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 3, 2, 1, 1, 1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1, 1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1, 1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

        # vgg19中MaxPool2d所在的层
        self.layers = {"4": "maxpool_1",
                       "9": "maxpool_2",
                       "18": "maxpool_3",
                       "27": "maxpool_4",
                       "36": "maxpool_5"}

    def forward(self, x):
        output = {}
        for name, layer in self.base_model._modules.items():
            # print(name, layer)
            # 从第一层开始获取图像的特征
            x = layer(x)

            # 如果是layers参数指定的特征，那就保存到output中
            if name in self.layers:
                output[self.layers[name]] = x

        x5 = output["maxpool_5"]  # size=(N, 512, x.H/32, x.W/32)
        x4 = output["maxpool_4"]  # size=(N, 512, x.H/16, x.W/16)
        x3 = output["maxpool_3"]  # size=(N, 256, x.H/8,  x.W/8)

        # 对特征进行相关的转置卷积操作,逐渐将图像放大到原始图像大小

        score = self.relu(self.deconv1(x5))  # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)  # 对应的元素相加, size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(score))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)  # 对应的元素相加, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)  # 最后一层是卷积，把输出变成分类的维数
        return score  # size=(N, n_class, x.H/1, x.W/1)


class FCN8s_VGG16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # num_classes:训练数据的类别
        self.num_classes = num_classes
        model = vgg16(pretrained=True)
        self.base_model = model.features

        # 定义几个需要的层操作，并且使用转置卷积将特征映射进行升维
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

        # vgg16中MaxPool2d所在的层
        self.layers = {"4": "maxpool_1",
                       "9": "maxpool_2",
                       "16": "maxpool_3",
                       "23": "maxpool_4",
                       "30": "maxpool_5"}

    def forward(self, x):
        output = {}
        for name, layer in self.base_model._modules.items():
            # print(name, layer)  # the number of layers and corresponding layer
            # get feature image from the first layer
            x = layer(x)

            # save output if layers parameters is given features
            if name in self.layers:
                output[self.layers[name]] = x

        x5 = output["maxpool_5"]  # size=(N, 512, x.H/32, x.W/32)
        x4 = output["maxpool_4"]  # size=(N, 512, x.H/16, x.W/16)
        x3 = output["maxpool_3"]  # size=(N, 256, x.H/8,  x.W/8)

        # transpose2d for features and upsample images to original image
        score = self.relu(self.deconv1(x5))  # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)  # element-wise addation, size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(score))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)  # element-wise addation, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)  # conv1x1 are used in the last layer
        return score  # size=(N, n_class, x.H/1, x.W/1)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    start_lr = 0.01
    ratio = 0.1
    lr = start_lr * (ratio ** (epoch // 100))
    for param_group in optimizer.param_groups:
        # print('lr : ', param_group['lr'])
        param_group['lr'] = lr
        # print('reduce to ', param_group['lr'])
    return lr


class FCN8s_ResNet(nn.Module):
    def __init__(self, num_classes, backbone="resnet34"):
        super(FCN8s_ResNet, self).__init__()
        self.num_classes = num_classes
        if "34" in backbone:
            pretrained_net = resnet34(pretrained=True)
            expansion = 1
        elif "18" in backbone:
            pretrained_net = resnet18(pretrained=True)
            expansion = 1
        elif "50" in backbone:
            pretrained_net = resnet50(pretrained=True)
            expansion = 4
        elif "101" in backbone:
            pretrained_net = resnet101(pretrained=True)
            expansion = 4
        else:
            assert "Please set correct backbone!"

        # stage1 channels: 128, output: 28x28
        self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-4])
        # stage2 channels: 256, output: 14x14
        self.stage2 = list(pretrained_net.children())[-4]
        # stage3 channels: 512, output: 7x7
        self.stage3 = list(pretrained_net.children())[-3]

        # three conv1x1 to fuse all channels information
        self.scores1 = nn.Conv2d(512 * expansion, num_classes, 1)
        self.scores2 = nn.Conv2d(256 * expansion, num_classes, 1)
        self.scores3 = nn.Conv2d(128 * expansion, num_classes, 1)

        # upsample 8x
        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16)  # 使用双线性 kernel
        # upsample 2x
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4)  # 使用双线性 kernel
        # upsample 2x
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4)  # 使用双线性 kernel

    def forward(self, x):
        x = self.stage1(x)
        s1 = x  # 224/8 = 28
        # print("s1: ", s1.shape)

        x = self.stage2(x)
        s2 = x  # 224/16 = 14
        # print("s2: ", s2.shape)

        x = self.stage3(x)
        s3 = x  # 224/32 = 7
        # print("s3: ", s3.shape)

        s3 = self.scores1(s3)  # fuse channels information
        s3 = self.upsample_2x(s3)  # upsample 2x size: 14x14
        # print("s3up: ", s3.shape)
        s2 = self.scores2(s2)
        s2 = s2 + s3  # 14*14

        s1 = self.scores3(s1)
        s2 = self.upsample_4x(s2)  # upsample 2x size: 28x28
        s = s1 + s2  # 28*28
        # print("s2up: ", s.shape)

        s = self.upsample_8x(s2)  # upsample 8x size: 224x224
        # print("s1up: ", s.shape)
        return s  # return feature


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(np.array(weight))


if __name__ == '__main__':
    # test for vgg16
    # net = FCN8s_VGG16(num_classes=21)
    # from torchsummary import summary
    # summary(net, (3, 512, 512), device="cpu")

    # net = resnet50(num_classes=21)
    # from torchsummary import summary
    # summary(net, (3, 512, 512), device="cpu")
    net = FCN8s_ResNet(num_classes=21, backbone="resnet50")
    x = torch.randn((1, 3, 512, 512))
    y = net(x)
    print(y.shape)
