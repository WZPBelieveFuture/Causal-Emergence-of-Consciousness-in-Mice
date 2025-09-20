# Let's load some other packages we need
import sys
import os
import pandas as pd
import random
import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
# import models
from src.models import InvertibleNN, Parellel_Renorm_Dynamic, train_and_memorize
from src.EI_calculation import approx_ei
from tqdm import tqdm
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
#Whether to use cuda or not
import argparse
parser = argparse.ArgumentParser(description='Causal Emergence of Mice')
# general settings
parser.add_argument('--cuda', type=int, default=15)
parser.add_argument('--stage', type=str, default='stage1')
parser.add_argument('--folder_name', type=str, default='des')
parser.add_argument('--mice_id', type=int, default=34)
parser.add_argument('--data_name', type=str, default='data')
parser.add_argument('--weight_id', type=int, default=0)
parser.add_argument('--epoch', type=int, default=3)
parser.add_argument('--version', type=int, default=0)
args = parser.parse_args()
device = torch.device('cuda:%s'%args.cuda)

train_input_list, train_target_list, test_input_list, test_target_list = [], [], [], []
mice_id = args.mice_id
train_input = pd.read_csv('./loc_data_allstage/%s/%s/%s/train_input.csv'%(args.folder_name, mice_id, args.stage), header=None)
train_input_list.append(np.array(train_input))
train_target = pd.read_csv('./loc_data_allstage/%s/%s/%s/train_target.csv'%(args.folder_name, mice_id, args.stage), header=None)
train_target_list.append(np.array(train_target))
test_input = pd.read_csv('./loc_data_allstage/%s/%s/%s/test_input.csv'%(args.folder_name, mice_id, args.stage), header=None)
test_input_list.append(np.array(test_input))
test_target = pd.read_csv('./loc_data_allstage/%s/%s/%s/test_target.csv'%(args.folder_name, mice_id, args.stage), header=None)
test_target_list.append(np.array(test_target))
train_input_list = np.concatenate(np.array(train_input_list),axis=0)
train_target_list = np.concatenate(np.array(train_target_list),axis=0)
test_input_list = np.concatenate(np.array(test_input_list),axis=0)
test_target_list = np.concatenate(np.array(test_target_list),axis=0)

train_input_data = torch.tensor(train_input_list, dtype=torch.float32) # [1,1000,64]
train_target_data = torch.tensor(train_target_list, dtype=torch.float32) # [1,1000,64]
test_input_data = torch.tensor(test_input_list, dtype=torch.float32) # [1,1000,64]
test_target_data = torch.tensor(test_target_list, dtype=torch.float32) # [1,1000,64]

EIs,CEs=train_and_memorize(train_input_data,train_target_data,test_input_data,test_target_data, args.weight_id, args.folder_name,args.mice_id,args.stage,'k-means',device=device, epoches=args.epoch,hidden_units=100,scale=100,batch_size=512,version=args.version)
