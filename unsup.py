from __future__ import division, print_function
from typing import Dict, SupportsRound, Tuple, Any
from os import PathLike
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch,gc
from torch.autograd import grad
from torch.autograd import Variable
import torch.fft ############### Pytorch >= 1.8.0
import torch.nn.functional as F
import SimpleITK as sitk
import os, glob
import json
import subprocess
import sys
from PIL import Image
import torch.nn.functional as nnf
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from easydict import EasyDict as edict
import random
import yaml
# currentdir = os.path.dirname(os.path.realpath(__file__))
# parentdir = os.path.dirname(currentdir)
# sys.path.append(parentdir)

if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"

################Parameter Loading#######################
def read_yaml(path):
    try:
        with open(path, 'r') as f:
            file = edict(yaml.load(f, Loader=yaml.FullLoader))
        return file
    except:
        print('NO FILE READ!')
        return None
para = read_yaml('./parameters.yml')

xDim = para.data.x 
yDim = para.data.y
zDim = para.data.z

def loss_Reg(y_pred):
        # For 3D reg
        # dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        # dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        # dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])
        # dy = dy * dy
        # dx = dx * dx
        # dz = dz * dz
        # d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        # grad = d / 3.0

        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        dy = dy * dy
        dx = dx * dx
        d = torch.mean(dx) + torch.mean(dy) 
        grad = d / 2.0
        return grad

##################Data Loading##########################
readfilename = './2D_Bulleye_full/train/train' + '.json' #### Read the json file from the 2DBrainfull_normalized
datapath = './2D_Bulleye_full/'
data = json.load(open(readfilename, 'r'))
outputs = []
keyword = 'train'
# outputs = np.array(outputs)

for i in range (0,len(data[keyword])):
    filename_src = datapath + data[keyword][i]['source']
    itkimage_src = sitk.ReadImage(filename_src)
    source_scan = sitk.GetArrayFromImage(itkimage_src)
    filename_tar = datapath + data[keyword][i]['target']
    itkimage_tar = sitk.ReadImage(filename_tar)
    target_scan = sitk.GetArrayFromImage(itkimage_tar)
    pair = np.concatenate((source_scan, target_scan), axis=0)
    outputs.append(pair)

train = torch.FloatTensor(outputs)
print (train.shape)

'''Check initilization'''
from losses import MSE, Grad
#################Network optimization########################
from networks import DiffeoDense  
net = []
for i in range(3):
    temp = DiffeoDense(inshape = (xDim,yDim),
				 nb_unet_features= [[16, 32],[ 32, 32, 16, 16]],
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res= True)
    net.append(temp)
net = net[0].to(dev)
# print (net)

valloader = torch.utils.data.DataLoader(train, batch_size = para.solver.batch_size, shuffle=True, num_workers=1)

running_loss = 0 
running_loss_val = 0
template_loss = 0
printfreq = 1
sigma = 0.02
repara_trick = 0.0
loss_array = torch.FloatTensor(para.solver.epochs,1).fill_(0)
loss_array_val = torch.FloatTensor(para.solver.epochs,1).fill_(0)


gradv_batch = torch.cuda.FloatTensor(para.solver.batch_size, 3, xDim, yDim).fill_(0).contiguous()
defIm_batch = torch.cuda.FloatTensor(para.solver.batch_size, 1, xDim, yDim).fill_(0).contiguous()
temp = torch.cuda.FloatTensor(para.solver.batch_size, 3, xDim, yDim).fill_(0).contiguous()
transformations = torch.cuda.FloatTensor(para.solver.batch_size, 3, xDim, yDim).fill_(0).contiguous() 
atlas = torch.cuda.FloatTensor(1, 1, xDim, yDim).fill_(0).contiguous()
atlas.requires_grad=True

gradv_batch_val = torch.cuda.FloatTensor(1, 3, xDim, yDim).fill_(0).contiguous()
defIm_batch_val = torch.cuda.FloatTensor(1, 1, xDim, yDim).fill_(0).contiguous() 
temp_val = torch.cuda.FloatTensor(1, 3, xDim, yDim).fill_(0).contiguous()
deform_size = [1, xDim, yDim]

if(para.model.loss == 'L2'):
    criterion = nn.MSELoss()
elif (para.model.loss == 'L1'):
    criterion = nn.L1Loss()
if(para.model.optimizer == 'Adam'):
    optimizer = optim.Adam(net.parameters(), lr= para.solver.lr)
elif (para.model.optimizer == 'SGD'):
    optimizer = optim.SGD(net.parameters(), lr= para.solver.lr, momentum=0.9)
if (para.model.scheduler == 'CosAn'):
    scheduler = CosineAnnealingLR(optimizer, T_max=len(valloader), eta_min=0)

optimizer_template = optim.Adam(net.parameters(), lr= para.solver.lr)
scheduler_template = CosineAnnealingLR(optimizer_template, T_max=len(valloader), eta_min=0)

# ##################Training###################################
for epoch in range(para.solver.epochs):
    total= 0; 
    total_val = 0; 
    total_template = 0; 
    net.train()
    print('epoch:', epoch)
    for j, image_data in enumerate(valloader):
        inputs = image_data.to(dev)
        b, c, w, h = inputs.shape
        optimizer.zero_grad()
        src_bch = inputs[:,0,...].reshape(b,1,w,h)
        tar_bch = inputs[:,1,...].reshape(b,1,w,h)
        pred = net(src_bch, tar_bch, registration = True)     
        loss = criterion(pred[0], tar_bch) 
        loss2 = np.linalg.norm(pred[3])
        loss_total = loss + 0.1*loss2   ### tune 0.1 with different values
        loss_total.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss_total.item()
        # print('[%d, %5d] loss: %.3f' %
        #     (epoch + 1, i + 1, running_loss ))
        total += running_loss
        running_loss = 0.0
    print ('total training loss:', total)