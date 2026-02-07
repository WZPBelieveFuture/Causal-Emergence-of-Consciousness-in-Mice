# Let's load some other packages we need
import sys
import os
import pandas as pd
import random
import torch
from torch import nn
from torch.nn.parameter import Parameter
from src.models import train_and_memorize
from tqdm import tqdm
import numpy as np
# use_cuda = torch.cuda.is_available()
import argparse 
parser = argparse.ArgumentParser(description='Causal Emergence of Mice')
# general settings 
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--stage', type=str, default='stage1')
parser.add_argument('--folder_name', type=str, default='des')
parser.add_argument('--mice_id', type=int, default=16)
parser.add_argument('--data_name', type=str, default='data')
parser.add_argument('--scale_id', type=int, default=0)
parser.add_argument('--ref_scale', type=int, default=0)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--epoch', type=int, default=100000)
parser.add_argument('--train_stage', type=int, default=1)
args = parser.parse_args()  
device = torch.device('cuda:%s'%args.cuda)  

train_input = pd.read_csv('./loc_data_allstage/%s/%s/%s/train_input.csv'%(args.folder_name, args.mice_id, args.stage), header=None)
train_input = np.array(train_input)
train_target = pd.read_csv('./loc_data_allstage/%s/%s/%s/train_target.csv'%(args.folder_name, args.mice_id, args.stage), header=None)
train_target = np.array(train_target)
test_input = pd.read_csv('./loc_data_allstage/%s/%s/%s/test_input.csv'%(args.folder_name, args.mice_id, args.stage), header=None)
test_input = np.array(test_input)
test_target = pd.read_csv('./loc_data_allstage/%s/%s/%s/test_target.csv'%(args.folder_name, args.mice_id, args.stage), header=None)
test_target = np.array(test_target)

train_input_data = torch.tensor(train_input, dtype=torch.float32)
train_target_data = torch.tensor(train_target, dtype=torch.float32)
test_input_data = torch.tensor(test_input, dtype=torch.float32)
test_target_data = torch.tensor(test_target, dtype=torch.float32)

train_and_memorize(train_input_data, train_target_data, test_input_data, test_target_data, args.scale_id, args.ref_scale, args.learning_rate, args.folder_name, args.mice_id, args.stage, 'k-means', device=device, epoches=args.epoch, hidden_units=50, min_dim=1, batch_size=512, train_stage=args.train_stage)
