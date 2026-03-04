import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from sklearn.model_selection import train_test_split
import os
import cv2
from torchvision.transforms import transforms,autoaugment
from tqdm import tqdm
import random
from PIL import Image
import matplotlib.pyplot as plt

HW = 224
# 变量结构：
# imagenet_norm 是一个包含两个子列表的列表：
# 第一个子列表 [0.485, 0.456, 0.406]：代表 RGB 三个通道的均值（mean），分别对应 Red、Green、Blue 通道
# 第二个子列表 [0.229, 0.224, 0.225]：代表 RGB 三个通道的标准差（std），同样对应 R、G、B 通道
# 数值来源：
# 这些数值是通过计算 ImageNet 数据集（包含超过 100 万张自然图像）中所有图像的像素值（归一化到 0-1 范围后）的均值和标准差得到的，是行业通用的标准值
imagenet_norm = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]

test_transform = transforms.Compose([
    transforms.ToTensor(),
])              # 测试集只需要转为张量
# 1. 图片是什么？
# 你平时看到的图片：
# 电脑里是 .jpg / .png
# Python 里打开是 PIL 图像
# 用 OpenCV 读是 numpy 数组
# 这些都是给人看、给文件存的格式

# 但是神经网络只认张量，所以必须转为张量参与训练

train_transform = transforms.Compose([
    transforms.ToPILImage(), # PIL图像对象，是一种专门用于图像操作的 Python 数据类型，区别于普通的 NumPy 数组或 PyTorch 张量
    transforms.RandomResizedCrop(HW), # 随机剪裁至224*224，每张都是原图的不同局部区域
    transforms.RandomHorizontalFlip(), # 水平翻转
    autoaugment.AutoAugment(),
    transforms.ToTensor(),
    # transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])                   # 训练集需要做各种变换。   效果参见https://pytorch.org/vision/stable/transforms.html



class foodDataset_global(Dataset):                      #数据集三要素： init ， getitem ， len
    def __init__(self, path, mode):
        y = None
        self.transform = None
        self.mode = mode

        pathDict = {'train':'training/labeled','train_unl':'training/unlabeled', 'val':'validation', 'test':'testing'}
        imgPaths = path +'/'+ pathDict[mode]                       # 根据模式定义对应数据集的路径

    # 不同模式的数据有不同的数据集和对应的数据增强方式
        if mode == 'test':
            x = self._readfile(imgPaths,label=False) # 是自定义的读文件函数！
            self.transform = test_transform                         #从文件读数据,测试机和无标签数据没有标签， trans方式也不一样
        elif mode == 'train':
            x, y =self._readfile(imgPaths,label=True)
            self.transform = train_transform
        elif mode == 'val':
            x, y =self._readfile(imgPaths,label=True)
            self.transform = test_transform
        elif mode == 'train_unl': # 原来在这里就区分了训练集是否有标签
            x = self._readfile(imgPaths,label=False)
            self.transform = train_transform

        if y is not None:                                    # 注意， 分类的标签必须转换为长整型： int64.
            y = torch.LongTensor(y)
        self.x, self.y = x, y # 以便该类的其他模块能够使用

    def __getitem__(self, index):                        # getitem 用于根据标签取数据,并把数据集取出来的数组转为张量，以便输入模型
        orix = self.x[index]    # 说白了就是根据索引index，即维度序号，范围为0-3取数据           # 取index的图片

        if self.transform == None:
            xT = torch.tensor(orix).float()  # 无自定义预处理：直接把原始像素数组转成浮点型张量，全局特征分支
        else:
            xT = self.transform(orix)  # 局部特征分支，作10次循环返回10张随机剪裁后的局部特征图片

        if self.y is not None:                       # 有标签， 则需要返回标签。 这里额外返回了原图orix， 方便后面画图。
            y = self.y[index]
            return xT, y, orix # 注意与读文件的函数不同，数据集返回的数据是张量形式
        else:
            return xT, orix

    def _readfile(self,path, label=True):                   # 定义一个读文件的函数
        if label:                                             # 有无标签， 文件结构是不一样的。
            x, y = [], [] # 空列表
            for i in tqdm(range(11)):    # tqdm是实时显示进度，方便判断速度                       # 有11类
                label = '/%02d/'%i    # 11个类别                             # %02必须为两位。 符合文件夹名字
                imgDirpath = path+label
                imglist = os.listdir(imgDirpath)         # listdir 可以列出文件夹下所有文件,即该类别下的所以.jpg图片文件
                xi = np.zeros((len(imglist), HW, HW, 3), dtype=np.uint8) #  8 位无符号整数,图像像素值天然是 0-255 的整数，用 uint8 存储最节省内存
                yi = np.zeros((len(imglist)), dtype=np.uint8)           # 先把放数据的格子打好。 x的维度是 照片数量*H*W*3
                                                            # y的维度是每张图片预测的类别值，和图片数量一样
                for j, each in enumerate(imglist):
                    imgpath = imgDirpath + each # 这才是真正的.jpg图片的实际路径
                    img = Image.open(imgpath)                  # 用image函数读入照片， 并且resize。
                    img = img.resize((HW, HW)) # 此时的img是一个PIL图像对象，完整、无随机裁剪的 224×224 原始图，进入全局特征分支！
                    xi[j,...] = img   # ...是numpy数组的省略索引，这里自动把img这个PIL对象转为图片的像素值放入xi  #在第j个位置放上数据和标签。
    # 即 xi[j, h, w, 0] 是第 j 张图第 h 行第 w 列的 R 值，xi[j, h, w, 1] 是 G 值，xi[j, h, w, 2] 是 B 值
    # 这里的xi 是一个4维数组！
                    yi[j] = i #内层for循环统计的是同一类别i下共有多少张图片，并把它们的索引和像素值放入4维数组xi中
                              # 所以类别都是序号i

        # x 是 4 维数组：维度顺序为 (总图像数量, 高度HW, 宽度HW, 通道数3)；
        # y 是 1 维数组：维度为 (总图像数量,)（只有 “总图像数量” 这一个维度）
                if i == 0:
                    x = xi
                    y = yi
                else:
        # axis=0 是按 “第一个维度，即图片数量维度” 拼接，不会改变单张图像的维度（H/W/C）
                    x = np.concatenate((x, xi), axis=0)    # 将11个文件夹的数据按类别（同类别按序号）合在一起。
                    y = np.concatenate((y, yi), axis=0)    # 按图片数量顺序合在一起，这样x和y，同一张图片的像素值和预测类别对应起来了
            print('读入有标签数据%d个 '%len(x)) # len(x) 会返回 x 这个 4 维数组第一个维度的长度（也就是总图像数）
            return x, y  # 把有标签的数据集加载并返回了
        else: # 无标签数据
        # 无标签数据本身没有人工标注的类别标签，所以在处理时会被统一赋予一个 “占位标签”（比如这里的00），看起来像 “只有一个类别“
            imgDirpath = path + '/00/'
            imgList = os.listdir(imgDirpath)
            x = np.zeros((len(imgList), HW, HW ,3),dtype=np.uint8)
            for i, each in enumerate(imgList):
                imgpath = imgDirpath + each # 具体的.jpg文件路径
                img = Image.open(imgpath)
                img = img.resize((HW, HW))
                x[i,...] = img # 仍然按照图片数量（序号），高、宽、通道 4个维度放入4维数组x
            return x # 无标签数据，只需要返回x

    def __len__(self):                      # len函数 负责返回长度。
        return len(self.x)

# 局部特征的数据集
class foodDataset_local(Dataset):                      #数据集三要素： init ， getitem ， len
    def __init__(self, path, mode):
        y = None
        self.transform = None
        self.mode = mode

        pathDict = {'train':'training/labeled','train_unl':'training/unlabeled', 'val':'validation', 'test':'testing'}
        imgPaths = path +'/'+ pathDict[mode]                       # 根据模式定义对应数据集的路径

    # 不同模式的数据有不同的数据集和对应的数据增强方式
        if mode == 'test':
            x = self._readfile(imgPaths,label=False) # 是自定义的读文件函数！
            self.transform = test_transform                         #从文件读数据,测试机和无标签数据没有标签， trans方式也不一样
        elif mode == 'train':
            x, y =self._readfile(imgPaths,label=True)
            self.transform = train_transform
        elif mode == 'val':
            x, y =self._readfile(imgPaths,label=True)
            self.transform = test_transform
        elif mode == 'train_unl': # 原来在这里就区分了训练集是否有标签
            x = self._readfile(imgPaths,label=False)
            self.transform = train_transform

        if y is not None:                                    # 注意， 分类的标签必须转换为长整型： int64.
            y = torch.LongTensor(y)
        self.x, self.y = x, y # 以便该类的其他模块能够使用

    def __getitem__(self, index):                        # getitem 用于根据标签取数据,并把数据集取出来的数组转为张量，以便输入模型
        orix = self.x[index]    # 说白了就是根据索引index，即维度序号，范围为0-3取数据           # 取index的图片

        # if self.transform == None:
        #     xT = torch.tensor(orix).float()  # 无自定义预处理：直接把原始像素数组转成浮点型张量，全局特征分支
        # else:
        local_imgs = []

        for i in range(10):
            xT = self.transform(orix)  # 局部特征分支，作10次循环返回10张随机剪裁后的局部特征图片
            local_imgs.append(xT)
        xT = torch.stack(local_imgs, dim=0)

        if self.y is not None:                       # 有标签， 则需要返回标签。 这里额外返回了原图orix， 方便后面画图。
            y = self.y[index]
            return xT, y, orix # 注意与读文件的函数不同，数据集返回的数据是张量形式
        else:
            return xT, orix

    def _readfile(self,path, label=True):                   # 定义一个读文件的函数
        if label:                                             # 有无标签， 文件结构是不一样的。
            x, y = [], [] # 空列表
            for i in tqdm(range(11)):    # tqdm是实时显示进度，方便判断速度                       # 有11类
                label = '/%02d/'%i    # 11个类别                             # %02必须为两位。 符合文件夹名字
                imgDirpath = path+label
                imglist = os.listdir(imgDirpath)         # listdir 可以列出文件夹下所有文件,即该类别下的所以.jpg图片文件
                xi = np.zeros((len(imglist), HW, HW, 3), dtype=np.uint8) #  8 位无符号整数,图像像素值天然是 0-255 的整数，用 uint8 存储最节省内存
                yi = np.zeros((len(imglist)), dtype=np.uint8)           # 先把放数据的格子打好。 x的维度是 照片数量*H*W*3
                                                            # y的维度是每张图片预测的类别值，和图片数量一样
                for j, each in enumerate(imglist):
                    imgpath = imgDirpath + each # 这才是真正的.jpg图片的实际路径
                    img = Image.open(imgpath)                  # 用image函数读入照片， 并且resize。
                    img = img.resize((HW, HW)) # 此时的img是一个PIL图像对象，完整、无随机裁剪的 224×224 原始图，进入全局特征分支！
                    xi[j,...] = img   # ...是numpy数组的省略索引，这里自动把img这个PIL对象转为图片的像素值放入xi  #在第j个位置放上数据和标签。
    # 即 xi[j, h, w, 0] 是第 j 张图第 h 行第 w 列的 R 值，xi[j, h, w, 1] 是 G 值，xi[j, h, w, 2] 是 B 值
    # 这里的xi 是一个4维数组！
                    yi[j] = i #内层for循环统计的是同一类别i下共有多少张图片，并把它们的索引和像素值放入4维数组xi中
                              # 所以类别都是序号i

        # x 是 4 维数组：维度顺序为 (总图像数量, 高度HW, 宽度HW, 通道数3)；
        # y 是 1 维数组：维度为 (总图像数量,)（只有 “总图像数量” 这一个维度）
                if i == 0:
                    x = xi
                    y = yi
                else:
        # axis=0 是按 “第一个维度，即图片数量维度” 拼接，不会改变单张图像的维度（H/W/C）
                    x = np.concatenate((x, xi), axis=0)    # 将11个文件夹的数据按类别（同类别按序号）合在一起。
                    y = np.concatenate((y, yi), axis=0)    # 按图片数量顺序合在一起，这样x和y，同一张图片的像素值和预测类别对应起来了
            print('读入有标签数据%d个 '%len(x)) # len(x) 会返回 x 这个 4 维数组第一个维度的长度（也就是总图像数）
            return x, y  # 把有标签的数据集加载并返回了
        else: # 无标签数据
        # 无标签数据本身没有人工标注的类别标签，所以在处理时会被统一赋予一个 “占位标签”（比如这里的00），看起来像 “只有一个类别“
            imgDirpath = path + '/00/'
            imgList = os.listdir(imgDirpath)
            x = np.zeros((len(imgList), HW, HW ,3),dtype=np.uint8)
            for i, each in enumerate(imgList):
                imgpath = imgDirpath + each # 具体的.jpg文件路径
                img = Image.open(imgpath)
                img = img.resize((HW, HW))
                x[i,...] = img # 仍然按照图片数量（序号），高、宽、通道 4个维度放入4维数组x
            return x # 无标签数据，只需要返回x

    def __len__(self):                      # len函数 负责返回长度。
        return len(self.x)









# 利用无标签数据和半监督学习产生的新数据集！
class noLabDataset(Dataset): # 为什么还要专门另写一个noLabDataset数据集?
                    # 答：这种 “区分” 的核心是：同一时刻，一个普通数据集实例只能加载「有标签」或「无标签」其中一种数据
                    # 半监督学习的训练逻辑要求：同一个批次里，必须同时包含有标签数据和无标签数据
    def __init__(self,dataloader, model, device, thres=0.85):
        super(noLabDataset, self).__init__()
        self.model = model      #模型也要传入进来
        self.device = device
        self.thres = thres      #这里置信度阈值 我设置的 0.85
        x, y = self._model_pred(dataloader)        #核心， 获得新的训练数据
        if x == []:                            # 如果没有， 就不启用这个数据集
            self.flag = False
        else:
            self.flag = True
            self.x = np.array(x)
            self.y = torch.LongTensor(y)
        # self.x = np.concatenate((np.array(x), train_dataset.x),axis=0)
        # self.y = torch.cat(((torch.LongTensor(y),train_dataset.y)),dim=0)
        self.transformers = train_transform

    def _model_pred(self, dataloader):  # 取的是train下的无标签数据！
        model = self.model
        device = self.device
        thres = self.thres
        pred_probs = []
        labels = []
        x = []
        y = []
        with torch.no_grad():                                  # 不训练， 要关掉梯度
            for data in dataloader:                            # 取数据进来预测得到伪标签，如果大于置信度就当作真实标签，返回x,y
                imgs = data[0].to(device)
                pred = model(imgs)                              #预测
                soft = torch.nn.Softmax(dim=1)             #softmax 可以返回一个概率分布
                pred_p = soft(pred)
                pred_max, preds = pred_p.max(1)          #得到最大值 ，和最大值的位置 。 就是置信度和标签。
                pred_probs.extend(pred_max.cpu().numpy().tolist())
                labels.extend(preds.cpu().numpy().tolist())        #把置信度和标签装起来

        for index, prob in enumerate(pred_probs):
            if prob > thres:                                  #如果置信度超过阈值， 就转化为可信的训练数据
                x.append(dataloader.dataset[index][1])
                y.append(labels[index])
        return x, y

    def __getitem__(self, index):                          # getitem 和len
        x = self.x[index]
        x= self.transformers(x)
        y = self.y[index]
        return x, y

    def __len__(self):
        return len(self.x)

def get_semi_loader(dataloader,model, device, thres):
    semi_set = noLabDataset(dataloader, model, device, thres)
    if semi_set.flag:                                                    #不可用时返回空
        dataloader = DataLoader(semi_set, batch_size=dataloader.batch_size,shuffle=True)
        return dataloader # 返回半监督学习产生的新数据集的迭代器
    else:
        return None


def global_getDataLoader(path, mode, batchSize):
    # 断言（强制检查）变量 mode 的值必须是列表 ['train', 'train_unl', 'val', 'test'] 中的其中一个；
    assert mode in ['train', 'train_unl', 'val', 'test']
    dataset = foodDataset_global(path, mode)
    if mode in ['test','train_unl']: # 无标签数据通常用于自监督学习 / 半监督学习，核心是利用数据本身的特征（如像素、纹理），而非标签，打乱的收益极低
        shuffle = False
    else: # 把有标签训练集、验证集打乱
        shuffle = True
    # 把 PyTorch 的 Dataset 看作 C++ 中的容器（比如 vector/map），DataLoader 看作访问这个容器的迭代器（Iterator）
    loader = DataLoader(dataset,batchSize,shuffle=shuffle)                      #装入loader
    return loader

def local_getDataLoader(path, mode, batchSize):
    # 断言（强制检查）变量 mode 的值必须是列表 ['train', 'train_unl', 'val', 'test'] 中的其中一个；
    assert mode in ['train', 'train_unl', 'val', 'test']
    dataset = foodDataset_local(path, mode)
    if mode in ['test','train_unl']: # 无标签数据通常用于自监督学习 / 半监督学习，核心是利用数据本身的特征（如像素、纹理），而非标签，打乱的收益极低
        shuffle = False
    else: # 把有标签训练集、验证集打乱
        shuffle = True
    # 把 PyTorch 的 Dataset 看作 C++ 中的容器（比如 vector/map），DataLoader 看作访问这个容器的迭代器（Iterator）
    loader = DataLoader(dataset,batchSize,shuffle=shuffle)                      #装入loader
    return loader



def samplePlot(dataset, isloader=True, isbat=False,ori=None):           #画图， 此函数不需要掌握。
    if isloader:
        dataset = dataset.dataset
    rows = 3
    cols = 3
    num = rows*cols
    # if isbat:
    #     dataset = dataset * 225
    datalen = len(dataset)
    randomNum = []
    while len(randomNum) < num:
        temp = random.randint(0,datalen-1)
        if temp not in randomNum:
            randomNum.append(temp)
    fig, axs = plt.subplots(nrows=rows,ncols=cols,squeeze=False)
    index = 0
    for i in range(rows):
        for j in range(cols):
            ax = axs[i, j]
            if isbat:
                ax.imshow(np.array(dataset[randomNum[index]].cpu().permute(1,2,0)))
            else:
                ax.imshow(np.array(dataset[randomNum[index]][0].cpu().permute(1,2,0)))
            index += 1
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.show()
    plt.tight_layout()
    if ori != None:
        fig2, axs2 = plt.subplots(nrows=rows,ncols=cols,squeeze=False)
        index = 0
        for i in range(rows):
            for j in range(cols):
                ax = axs2[i, j]
                if isbat:
                    ax.imshow(np.array(dataset[randomNum[index]][-1]))
                else:
                    ax.imshow(np.array(dataset[randomNum[index]][-1]))
                index += 1
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.show()
        plt.tight_layout()





if __name__ == '__main__':   #运行的模块，  如果你运行的模块是当前模块
    print("你运行的是data.py文件")
    filepath = '../food-11_sample'
    train_loader = getDataLoader(filepath, 'train', 8)
    for i in range(3):
        # 目的是快速验证训练集数据加载正常、图片无损坏，无需看太多
        samplePlot(train_loader,True,isbat=False,ori=True) # 自定义的可视化画图函数，不需要掌握
    val_loader = getDataLoader(filepath, 'val', 8)
    for i in range(100):
        samplePlot(val_loader,True,isbat=False,ori=True)
    ##########################

