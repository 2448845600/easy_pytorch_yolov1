import torch.nn as nn
from model.net import vgg16_bn

'''
VGG16的输入是224*224*3，YOLO的输入是448*448*3，所以vgg16_bn().features的输出是14*14*512，
故增加一层池化nn.AdaptiveAvgPool2d((7, 7))，YOLO最后的输出还是7*7*30
'''


class YOLO(nn.Module):
    def __init__(self, num_classes=20, init_weights=True):
        super(YOLO, self).__init__()
        self.basic_net = vgg16_bn().features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7)) # 自适应平均池化
        self.detector_head = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 7 * 7 * (2 * 5 + num_classes))
        )
        self.init_head()

    def init_head(self):
        for layer in self.detector_head:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(0, 0.01)
                layer.bias.data.zero_()

    def forward(self, input):
        x = self.basic_net(input)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        output = self.detector_head(x)
        return output


if __name__ == '__main__':
    yolo = YOLO()
    print(yolo)
