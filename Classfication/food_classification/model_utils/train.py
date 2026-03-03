from tqdm import tqdm
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from model_utils.data import samplePlot, get_semi_loader
from model_utils.model import SelfAttentionFusion
import torch.nn as nn


def train_val(para): # 就这一个函数

########################################################
    model = para['model']
    semi_loader = para['no_label_Loader']
    train_loader =para['train_loader']
    val_loader = para['val_loader']
    optimizer = para['optimizer']
    loss = para['loss']
    epoch = para['epoch']
    device = para['device']
    save_path = para['save_path']
    save_acc = para['save_acc']
    pre_path = para['pre_path']
    max_acc = para['max_acc']
    val_epoch = para['val_epoch']
    acc_thres = para['acc_thres']
    conf_thres = para['conf_thres']
    do_semi= para['do_semi']

    semi_epoch = 10
###################################################
    no_label_Loader = None
    if pre_path != None:
        model = torch.load(pre_path)
    model = model.to(device)
    selfattention_fusion = SelfAttentionFusion(dim=512).to(device)
    # model = torch.nn.DataParallel(model).to(device)
    # model.device_ids = [0,1]  多卡训练

    plt_train_loss = []
    plt_train_acc = []
    plt_val_loss = []
    plt_val_acc = []
    plt_semi_acc = []
    val_rel = []
    max_acc = 0

    for i in range(epoch):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        val_loss = 0.0
        semi_acc = 0.0

        # 用zip同步迭代两个loader，保证取到的是同一张图的全局/局部版本
        # for data_global, data_local in tqdm(zip(train_loader_global, train_loader_local)):
        for data in tqdm(train_loader):                    #取数据
            optimizer.zero_grad()                           # 梯度置0
            x, target = data[0].to(device), data[1].to(device)
            pred = model(x)                                 #模型前向
            pred_local = model(x)  # 局部特征
            pred_local_enhanced = selfattention_fusion(pred_local)  # 融合后的局部增强特征
            # 再拼接pred和pred_local_enhanced
            # 先把用resnet18得到的全局特征（维度为分类头11）升维到512(和融合后的局部特征同维)
            pred_all = nn.Linear(pred, 512).to(device)
            pred_final =  torch.cat([pred_all, pred_local_enhanced], dim=1)

            # 再通过展平，全连接，把pred_final变成分类头11维

            # 再用这个pred_final变成分类头后的结果与标签target计算loss，同时更新两个模型

            bat_loss = loss(pred_final, target)                   # 算交叉熵loss
            # 假设我自己的模型换成resnet来提取局部特征，只更新这一个共同的特征提取器
            bat_loss.backward()                                 # 回传梯度
            optimizer.step()                                    # 根据梯度更新
            train_loss += bat_loss.item()    #.detach 表示去掉梯度
            # axis=1 表示按行（第二个维度）找(最大值)的索引, data[1].numpy()：获取真实标签并转为 NumPy 数组
# 提示：
# PyTorch 的 DataLoader 加载数据集后，迭代返回的每一批数据，本质上就是 (x, y) 这种格式 ——x 对应模型输入（特征）
     # y 对应标签（监督信号），这是 PyTorch 处理监督学习任务的通用且标准的约定
    # 所以data[0]是x，data[1]是y
    # 记录分类预测正确的样本总数,表示准确率
            train_acc += np.sum(np.argmax(pred.cpu().data.numpy(),axis=1) == data[1].numpy())

            # 预测值和标签相等，正确数就加1.  相等多个， 就加几。

        # 开始取无标签数据进行半监督训练了！
# 半监督阶段和有监督阶段更新的是同一个模型的参数
        if no_label_Loader != None:
            for data in tqdm(no_label_Loader):
                optimizer.zero_grad()
                x , target = data[0].to(device), data[1].to(device)
                pred = model(x)
                bat_loss = loss(pred, target)
                bat_loss.backward()
                optimizer.step()

                semi_acc += np.sum(np.argmax(pred.cpu().data.numpy(),axis=1)== data[1].numpy())
            plt_semi_acc .append(semi_acc/no_label_Loader.dataset.__len__())
            print('semi_acc:', plt_semi_acc[-1])
        # 结束半监督训练语句块了

        plt_train_loss.append(train_loss/train_loader.dataset.__len__())
        plt_train_acc.append(train_acc/train_loader.dataset.__len__())
        if i % val_epoch == 0:  # 每隔 val_epoch 个迭代 /epoch，执行一次验证（validation）,保存更优模型操作
            model.eval() # 切换到验证模式
            with torch.no_grad(): # 验证模式不需要计算梯度
                for valdata in val_loader:
                    val_x , val_target = valdata[0].to(device), valdata[1].to(device)
                    val_pred = model(val_x)
                    val_bat_loss = loss(val_pred, val_target)
                    val_loss += val_bat_loss.cpu().item()

                    val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == valdata[1].numpy())
                    val_rel.append(val_pred)


            val_acc = val_acc/val_loader.dataset.__len__()
            if val_acc > max_acc:
                torch.save(model, save_path)
                max_acc = val_acc


            plt_val_loss.append(val_loss/val_loader.dataset.__len__())
            plt_val_acc.append(val_acc)
            print('[%03d/%03d] %2.2f sec(s) TrainAcc : %3.6f TrainLoss : %3.6f | valAcc: %3.6f valLoss: %3.6f  ' % \
                  (i, epoch, time.time()-start_time, plt_train_acc[-1], plt_train_loss[-1], plt_val_acc[-1], plt_val_loss[-1])
                  )
        else: # 复用上一次的验证指标（保证绘图数据长度一致,否则y的个数只有epoch/val_epoch个，与x的个数不对应）
            plt_val_loss.append(plt_val_loss[-1])
            plt_val_acc.append(plt_val_acc[-1])

        # 每间隔semi_epoch轮才执行一次半监督逻辑」，避免每一轮都处理无标签数据，节省计算资源

        if do_semi and plt_val_acc[-1] > acc_thres and i % semi_epoch==0:         # 如果启用半监督， 且精确度超过阈值， 则开始。
            no_label_Loader = get_semi_loader(semi_loader,  model, device, conf_thres)


    plt.plot(plt_train_loss)                   # 画图。
    plt.plot(plt_val_loss)
    plt.title('loss')
    plt.legend(['train', 'val'])
    plt.show()

    plt.plot(plt_train_acc)
    plt.plot(plt_val_acc)
    plt.title('Accuracy')
    plt.legend(['train', 'val'])
    plt.savefig('acc.png')
    plt.show()
