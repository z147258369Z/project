import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Block
# Block：ViT 的 “核心计算模块”，包含多头自注意力（MSA）、层归一化（LN）、前馈网络（FFN），是 ViT 实现特征提取的核心
import torchvision.models as models



def set_parameter_requires_grad(model, linear_probing):
    if linear_probing: # 如果是线性探测，只训练最后的分类头
        for param in model.parameters():
            param.requires_grad = False                             # 一个参数的requires_grad梯度设为false， 则训练时就会不更新

class MyModel(nn.Module):                  #自己的模型
    def __init__(self,numclass = 2):
        super(MyModel, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) # 池化操作，在尽量保留特征的核心信息下，1*1是最大化保留原特征了，降低图片尺寸，减少参数计算量，即降维
        )  #112*112
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )  #56*56
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )  #28*28
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )  #14*14

        # 这两个是全局池化和全连接把局部特征变成全局特征了，去掉！
        # self.pool1 = nn.MaxPool2d(2)#7*7
        # self.fc = nn.Linear(25088, 512)

        # self.drop = nn.Dropout(0.5)
        self.relu1 = nn.ReLU(inplace=True)
        # 不用这个来分类了，去掉
        # self.fc2 = nn.Linear(512, numclass)  # 得到分类头
    def forward(self,x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # 不用变成全局特征了，而是变成和vit分支输出的特征同维度，然后return回去

       # x = self.pool1(x)
# 这行代码的作用是将任意维度的张量（Tensor）“展平” 成二维张量：
        # 第一维保持不变（通常是批量大小 batch_size）；
        # 第二维自动计算，把剩下的所有维度合并成一维（通过 -1 让 PyTorch 自动推导）
        # x = x.view(x.size()[0],-1)       #view 类似于reshape  这里指定了第一维度为batch大小，第二维度为适应的，即剩多少， 就是多少维。
        #                                 # 这里就是将特征展平。  展为 B*N  ，N为特征维度,B为batchsize大小
        # x = self.fc(x)
        # # x = self.drop(x)
        # x = self.relu1(x)
        #
        # # 这里不输出分类头，而是直接把这个512维局部特征（或间接全局特征）return出去与vit输出的512维的全局关联特征作特征融合
        # # 再用融合后的特征输出分类头
        #
        # x = self.fc2(x) #这里得到了最后的分类结果

        #
        return x  # 输入没经过数据增强就是全局特征分支，经过增强就是局部特征分支

# 新增：自注意力融合模块（和MyModel同级）
class SelfAttentionFusion(nn.Module):  # 注意类名规范：驼峰命名，避免下划线
    def __init__(self, dim=512):
        super().__init__()
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)  # 只有一套q、k、v，是单头！
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, local_feat_map):
        batch_size, c, h, w = local_feat_map.shape
        # 特征图转序列：[batch, 512, 14, 14] → [batch, 196, 512]
        local_feat = local_feat_map.permute(0, 2, 3, 1).reshape(batch_size, h*w, c)
        # 映射Q/K/V
        Q = self.W_q(local_feat)
        K = self.W_k(local_feat)
        V = self.W_v(local_feat)
        # 计算注意力权重
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(c)
        attn_weights = self.softmax(attn_scores)
        # 加权融合得到局部增强特征
        local_enhanced = torch.matmul(attn_weights, V).sum(dim=1)
        return local_enhanced

# def model_Datapara(model, device,  pre_path=None):
#     model = torch.nn.DataParallel(model).to(device)
# 
#     model_dict = torch.load(pre_path).module.state_dict()
#     model.module.load_state_dict(model_dict)
#     return model

#传入模型名字，和分类数， 返回你想要的模型
def initialize_model(model_name, num_classes, linear_prob=False, use_pretrained=True):
    # 初始化将在此if语句中设置的这些变量。
    # 每个变量都是模型特定的。
    model_ft = None
    input_size = 0
    if model_name =="MyModel": # 用自己的模型
        if use_pretrained == True:
            model_ft = torch.load('model_save/MyModel')
        else:
            model_ft = MyModel(num_classes)
        input_size = 224

    elif model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)            # 从网络下载模型  pretrain true 使用参数和架构， false 仅使用架构。
        set_parameter_requires_grad(model_ft, linear_prob)            # 是否为线性探测，线性探测： 固定特征提取器不训练。
        num_ftrs = model_ft.fc.in_features  #分类头的输入维度
        model_ft.fc = nn.Linear(num_ftrs, num_classes)            # 删掉原来分类头， 更改最后一层为想要的分类数的分类头。
        input_size = 224
        
    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, linear_prob)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    elif model_name == "googlenet":
        """ googlenet
        """
        model_ft = models.googlenet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, linear_prob)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224


    elif model_name == "alexnet":
        """ Alexnet
 """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, linear_prob)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
 """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, linear_prob)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
 """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, linear_prob)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
 """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, linear_prob)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
 Be careful, expects (299,299) sized images and has auxiliary output
 """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, linear_prob)
        # 处理辅助网络
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # 处理主要网络
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model_utils name, exiting...")
        exit()

    return model_ft, input_size


def prilearn_para(model_ft,linear_prob):
    # 将模型发送到GPU
    device = torch.device("cuda:0")
    model_ft = model_ft.to(device)

    # 在此运行中收集要优化/更新的参数。
    # 如果我们正在进行微调，我们将更新所有参数，即非线性探测,特征层 + 分类层都更新
    # 但如果我们正在进行特征提取，我们只会更新刚刚初始化的参数，即`requires_grad`的参数为True。
    # 即线性探测，冻结预训练的特征提取层（只用来 “提取特征”），仅更新最后初始化的分类层参数
    # 因为特征提取的目标是提取特征，不需要训练特征提取的能力，微调才是训练特征提取的能力!
    params_to_update = model_ft.parameters()  # 先默认赋值为模型所有参数
    print("Params to learn:")
    if linear_prob:
        params_to_update = []  #  清空默认列表，重新筛选
        for name,param in model_ft.named_parameters():  # 遍历模型所有参数（带参数名）
            if param.requires_grad == True: # 只选要求训练的参数，然后再加入 params_to_update
                params_to_update.append(param)
                print("\t",name)
    else:  # 非线性探测
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    #
    # 打印出需要优化的参数的名字
    # optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)



     # 深度学习中模型参数初始化的标准实现
# 遍历深度学习模型（如 PyTorch 模型）中的所有层，针对卷积层（Conv）和批量归一化层（BatchNorm）分别采用不同的策略初始化权重和偏置参数
def init_para(model):
    def weights_init(model):
        classname = model.__class__.__name__  #   classname = model.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0)
 # PyTorch 模型的 apply() 方法会递归地将传入的函数应用到模型的每一个子模块（层）上，即遍历网络的各层，对不同层设置不同参数
# 这意味着即使你的模型包含嵌套的子网络（如 Sequential、自定义 Module），也能遍历到所有的 Conv/BatchNorm 层
    model.apply(weights_init)
    return model










