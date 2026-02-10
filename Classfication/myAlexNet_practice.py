
import  torch.nn as nn
import torchvision.models as models

alexnet=models.alexnet()

# print(alexnet)

class MyAlexNet_practice(nn.Module):
    # __init__ 是类的构造函数（也叫初始化方法），当你创建这个类的实例（比如 model = MyAlexNet()）时，__init__ 会被自动调用。
    def __init__(self):
        super(MyAlexNet_practice, self).__init__()
        self.relu = nn.ReLU() #直接使用官方模型的ReLU函数，不作修改
        self.drop = nn.Dropout(0.5)  #防止过拟合，随机失活50%神经元
#  卷积层的输入参数： 输入特征图数量 输出特征图 数量卷积核大小 步长 padding

        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 64, kernel_size=11, stride=4, padding=2)
        self.pool1 = nn.MaxPool2d(3, stride=2)  #每3个选一个最大的，步长为2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(3, stride=2)

        self.conv3 = nn.Conv2d(in_channels=192,out_channels=384,kernel_size=3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.conv5 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.pool3 = nn.MaxPool2d(3,stride=2)
        # 不论多少维，自适应统一成6*6
        self.adapool = nn.AdaptiveAvgPool2d(output_size=6)

        # 展平，全连接层 256*6*6
        self.fc1 = nn.Linear(9216,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,1000)

    # 它专门定义了数据在神经网络中的前向传播路径
    def forward(self,x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool3(x)  #特质提取结束

        # 自适应6*6
        x = self.adapool(x)

        x = x.view(x.size()[0], -1)

        # 全连接
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)

        return x

import torch

myalexnet = MyAlexNet_practice()

def get_parameter_number(model):
    # p.numel()：计算单个参数张量的元素个数
    # p是迭代器中的每一个参数张量（比如上面的卷积权重张量），numel()是PyTorch张量的方法，全称是「number ofelements」，
    # 作用是返回当前张量中元素的总个数（即参数数量）

    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"Total": total_num, "Trainable": trainable_num } #键值对

print(get_parameter_number(myalexnet))

# 创建一个形状为 (4, 3, 224, 224) 的张量，并且张量中所有元素的值都初始化为 0，
# 通常用来模拟深度学习中批量的图像数据（比如测试模型输入是否匹配、构造初始输入等）
image = torch.zeros(4,3,224,224)

out = myalexnet(image) #张量x作为模型的输入

print(out.size())