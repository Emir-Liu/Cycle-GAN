import os
import torch
import pickle
import random
import numpy as np

# 下面两个是用pickle模块来保存/获取数据
def save_file(file_path,file_name,data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path+file_name, "wb") as file:
        pickle.dump(data, file)

def load_file(file_path,file_name):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path+file_name, "rb") as file:
        data = pickle.load(file)
    return data

# 设计的随机种子，用于结果的复现，一般都用下面的函数，不用的可以注释掉
def set_seed(seed):
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #np.random.seed(seed)
    random.seed(seed)
'''
for name, param in self.named_parameters():
    if param.requires_grad:
        print(name)
'''
# model.named_parameters() 显示模型中的训练参数
def show_parameters(model_named_parameters):
    for name, param in model_named_parameters:
        if param.requires_grad:
            print(name)

# 用于保存神经网络的模型
'''
存储的结果为file_path + file_name + '.ep%d'%epoch
'''
def save_model(model,epoch,file_path,file_name):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    output_path = file_path + file_name + '.ep%d'%epoch
    torch.save(model.state_dict(),output_path)
    print("EP:%d Model Saved on:" % epoch, output_path)

def load_model(model,epoch,file_path,file_name):
    chkpt = torch.load(file_path + file_name + '.ep%d' % epoch)
    model.load_state_dict(chkpt)
