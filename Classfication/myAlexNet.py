import torchvision.models as models #引入官方的很多模型，可以直接用
import torch.nn as nn


alexnet = models.alexnet()


print(alexnet)

class MyAlexNet(nn.Module):
    # __init__ 是类的构造函数（也叫初始化方法），当你创建这个类的实例（比如 model = MyAlexNet()）时，__init__ 会被自动调用。
    def __init__(self):
        super(MyAlexNet, self).__init__()
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
# 输入特征图数量 输出特征图 数量卷积核大小 步长 padding
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2) #卷积
        self.pool1 = nn.MaxPool2d(3, stride=2)
        self.conv2 = nn.Conv2d(64,192,5,1,2)
        self.pool2 = nn.MaxPool2d(3, stride=2)

        self.conv3 = nn.Conv2d(192,384,3,1,1)

        self.conv4 = nn.Conv2d(384,256,3,1,1)

        self.conv5 = nn.Conv2d(256, 256, 3, 1, 1)

        self.pool3 = nn.MaxPool2d(3, stride=2)
        self.adapool = nn.AdaptiveAvgPool2d(output_size=6)

        self.fc1 = nn.Linear(9216,4096)

        self.fc2 = nn.Linear(4096,4096)

        self.fc3 = nn.Linear(4096,1000)

    def forward(self,x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)


        x = self.conv3(x)
        x = self.relu(x)
        print(x.size())
        x = self.conv4(x)
        x = self.relu(x)
        print(x.size())
        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool3(x)
        print(x.size())
        x= self.adapool(x)

        # 专门适配全连接层（nn.Linear）的输入要求（全连接层只接受二维张量，第一维是批次，第二维是特征数

        # x.size()[0]：获取张量x的第0维长度（也就是batch_size，比如一批有8张图，这就是8），展平后保留这个维度，保证批次信息不丢失
        #
        # -1：PyTorch中view方法的 “自动计算” 占位符，意思是 “根据其他维度的长度，自动算出这一维该是多少”，不用手动计算特征总数。
        #
        # x.view(...)：在不改变张量数据的前提下，重塑张量形状（类似reshape，但更贴合PyTorch动态图特性）。

        x = x.view(x.size()[0], -1) #用来展平、拉直

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)

        return x


import torch

myalexnet = MyAlexNet()

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters()) #返回模型的总参数量
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad) # 返回模型的可训练参数量
    return {'Total': total_num, 'Trainable': trainable_num}


print(get_parameter_number(myalexnet))

img = torch.zeros((4,3,224,224))

out = myalexnet(img)

print(out.size())

