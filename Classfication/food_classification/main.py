import random
import torch
import torch.nn as nn
import numpy as np
import os


from model_utils.model import initialize_model
from model_utils.train import train_val
from model_utils.data import global_getDataLoader
from model_utils.data import local_getDataLoader


# os.environ['CUDA_VISIBLE_DEVICES']='0,1'


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
#################################################################
seed_everything(0)
###############################################


model_name = 'resnet18'
##########################################

num_class = 11
batchSize = 32
learning_rate = 1e-4
loss = nn.CrossEntropyLoss()
epoch = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
##########################################
filepath = 'food-11_sample'
#filepath = 'food-11'
##########################

#读数据
global_train_loader = global_getDataLoader(filepath, 'train', batchSize)
local_train_loader =  local_getDataLoader(filepath, 'train', batchSize)

global_val_loader = global_getDataLoader(filepath, 'val', batchSize)
local_val_loader = local_getDataLoader(filepath, 'val', batchSize)


global_no_label_Loader = global_getDataLoader(filepath,'train_unl', batchSize)
local_no_label_Loader = local_getDataLoader(filepath,'train_unl', batchSize)

#模型和超参数, 超参数（Hyperparameter）是模型训练前手动设置、训练过程中不会被优化的参数
model, input_size = initialize_model(model_name, 11, use_pretrained=False)

print(input_size)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=1e-4)

save_path = 'model_save/model.pth'

trainpara = {
            "model" : model,
             'global_train_loader': global_train_loader,
             'local_train_loader': local_train_loader,
             'global_val_loader': global_val_loader,
             'local_val_loader': local_val_loader,
             'global_no_label_Loader': global_no_label_Loader,
             'local_no_label_Loader': local_no_label_Loader,
             'optimizer': optimizer,
            'batchSize': batchSize,
             'loss': loss,
             'epoch': epoch,
             'device': device,
             'save_path': save_path,
             'save_acc': True,
             'max_acc': 0.5,
             'val_epoch' : 1,
             'acc_thres' : 0.7,
             'conf_thres' : 0.99,
             'do_semi' : True,
            "pre_path" : None
             }


if __name__ == '__main__':  # 如果运行的模块是当前模块，才执行下面的代码
    train_val(trainpara)  # 这里直接调用的train.py中的train_val函数，在main.py中设置好总的trainpara传入