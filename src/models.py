import torch
import numpy as np
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
from src.EI_calculation import approx_ei
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
import matplotlib.pyplot as plt
import random,os
from copy import deepcopy
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class InvertibleNN(nn.Module):
    def __init__(self, nets, nett, mask, device):
        super(InvertibleNN, self).__init__()
        self.device = device
        self.mask = nn.Parameter(mask, requires_grad=False)
        length = mask.size()[0] // 2
        self.t = torch.nn.ModuleList([nett() for _ in range(length)])
        self.s = torch.nn.ModuleList([nets() for _ in range(length)])
        self.size = mask.size()[1]
    def g(self, z):
        x = z
        log_det_J = x.new_zeros(x.shape[0], device=self.device)
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
            log_det_J += s.sum(dim=1)
        return x, log_det_J

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0], device=self.device), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

class Parellel_Renorm_Dynamic(nn.Module):
    def __init__(self, sym_size, latent_size, effect_size, cut_size, hidden_units,normalized_state,device,is_random=False):
        super(Parellel_Renorm_Dynamic, self).__init__()
        if latent_size < 1 or latent_size > sym_size:
            print('Latent Size is too small(<1) or too large(>input_size):', latent_size)
            return
        
        self.device = device
        self.latent_size = latent_size
        self.effect_size = effect_size
        self.sym_size = sym_size
        i = sym_size
        flows = []
        dynamics_modules = []
        inverse_dynamics_modules = []
        while i > latent_size:
            input_size = max(latent_size, i)
            if i == sym_size:
                mid_size = sym_size
                if mid_size % 2 == 1:
                    mid_size = mid_size - 1
                dynamics = self.build_dynamics(mid_size, hidden_units)
                dynamics_modules.append(dynamics)
                inverse_dynamics = self.build_dynamics(mid_size, hidden_units)
                inverse_dynamics_modules.append(inverse_dynamics)
                flow = self.build_flow(mid_size, hidden_units)
                flows.append(flow)
            
            flow = self.build_flow(mid_size, hidden_units)
            flows.append(flow)
            mid_size = max(latent_size, int(i // cut_size))

            dynamics = self.build_dynamics(mid_size, hidden_units)
            dynamics_modules.append(dynamics)
            inverse_dynamics = self.build_dynamics(mid_size, hidden_units)
            inverse_dynamics_modules.append(inverse_dynamics)
            if mid_size % 2 == 1:
                mid_size = mid_size - 1
            i = mid_size

        self.flows = nn.ModuleList(flows)
        self.dynamics_modules = nn.ModuleList(dynamics_modules)
        self.inverse_dynamics_modules = nn.ModuleList(inverse_dynamics_modules)
        
        self.normalized_state=normalized_state
        self.is_random = is_random

    def build_flow(self, input_size, hidden_units):
        if input_size % 2 !=0 and input_size > 1:
            input_size = input_size - 1
        nets = lambda: nn.Sequential(nn.Linear(input_size, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, input_size), nn.Tanh())
        nett = lambda: nn.Sequential(nn.Linear(input_size, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, input_size))

        #Make masks
        mask1 = torch.cat((torch.zeros(1, input_size // 2, device=self.device), 
                           torch.ones(1, input_size // 2, device=self.device)), 1)
        mask2 = 1 - mask1
        masks = torch.cat((mask1, mask2, mask1, mask2, mask1, mask2,
                           mask1, mask2, mask1, mask2, mask1, mask2), 0)
        
        flow = InvertibleNN(nets, nett, masks, self.device)
        return flow
    
    def build_dynamics(self, mid_size, hidden_units):
        dynamics = nn.Sequential(nn.Linear(mid_size, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, hidden_units), nn.LeakyReLU(), 
                                     nn.Linear(hidden_units, mid_size))
        return dynamics
    
    def forward(self, x, delay=1):
        if len(x.size()) <= 1:
            x = x.unsqueeze(0)
        ss = self.encoding(x)
        s_nexts = []
        ys = []
        for i,s in enumerate(ss):
            for t in range(delay):
                s_next = self.dynamics_modules[i](s) + s
                s=s_next
            if self.normalized_state:
                s_next = torch.tanh(s_next)
            if self.is_random:
                s_next = s_next + torch.relu(self.sigmas.repeat(s_next.size()[0],1)) * torch.randn(s_next.size(), device=self.device)
            y = self.decoding(s_next, i)
            s_nexts.append(s_next)
            ys.append(y)
        return ys, ss, s_nexts
    
    def train_forward(self, x, weights,delay=1):
        if len(x.size())<=1:
            x = x.unsqueeze(0)
        ss = self.encoding1(x, weights)
        s_nexts = []
        ys = []
        i = len(ss)-1
        s = ss[i]
        s_next = self.dynamics_modules[i](s) + s
        if self.normalized_state:
            s_next = torch.tanh(s_next)
        y = self.decoding(s_next, i)
        s_nexts.append(s_next)
        ys.append(y)
        return ys, ss, s_nexts
    
    def train_back_forward(self, x, weights,delay=1):
        if len(x.size())<=1:
            x = x.unsqueeze(0)
        ss = self.encoding1(x,weights)
        s_nexts = []
        ys = []
        i = len(ss)-1
        s = ss[i]
        s_next = self.inverse_dynamics_modules[i](s) + s
        if self.is_random:
            s_next = s_next + torch.relu(self.sigmas.repeat(s_next.size()[0],1)) * torch.randn(s_next.size(),device=self.device)
        if self.normalized_state:
            s_next = torch.tanh(s_next)
        s_nexts.append(s_next)
        return ys, ss, s_nexts
        
    def encoding1(self, x, weight):
        xx = x
        if len(x.size()) > 1:
            if x.size()[1] < self.sym_size:
                xx = torch.cat((x, torch.zeros([x.size()[0], self.sym_size - x.size()[1]], device=self.device)), 1)
        else:
            if x.size()[0] < self.sym_size:
                xx = torch.cat((x, torch.zeros([self.sym_size - x.size()[0]], device=self.device)), 0)
        y = xx
        ys = []
        for i,flow in enumerate(self.flows):
            if y.size()[1] > flow.size:
                y = y[:, :flow.size]
            y,_ = flow.f(y)
            pdict = dict(self.dynamics_modules[i].named_parameters())
            lsize = pdict['0.weight'].size()[1]
            y = y[:, :lsize]
            ys.append(y)
            if i >= weight:
                break
        return ys

    def decoding(self, s_next, level):
        y = s_next
        for i in range(level+1)[::-1]:
            flow = self.flows[i]
            end_size = y.shape[1]
            sz = flow.size - end_size
            if sz > 0:
                noise = distributions.MultivariateNormal(torch.zeros(sz, device=self.device), torch.eye(sz, device=self.device)).sample((y.size()[0], 1))/3
                if y.size()[0]>1:
                    noise = noise.squeeze(1)
                else:
                    noise = noise.squeeze(0)
                y = torch.cat((y, noise), 1)
            y,_ = flow.g(y)
        return y
    
    def encoding(self, x):
        xx = x
        if len(x.size()) > 1:
            if x.size()[1] < self.sym_size:
                xx = torch.cat((x, torch.zeros([x.size()[0], self.sym_size - x.size()[1]], device=self.device)), 1)
        else:
            if x.size()[0] < self.sym_size:
                xx = torch.cat((x, torch.zeros([self.sym_size - x.size()[0]], device=self.device)), 0)
        y = xx
        ys = []
        for i,flow in enumerate(self.flows):
            if y.size()[1] > flow.size:
                y = y[:, :flow.size]
            y,_ = flow.f(y)

            pdict = dict(self.dynamics_modules[i].named_parameters())
            lsize = pdict['0.weight'].size()[1]
            y = y[:, :lsize]
            ys.append(y)
        return ys
    
    def loss(self, predictions, real, loss_f):
        losses = []
        sum_loss = 0
        for i, predict in enumerate(predictions):
            loss = loss_f(real, predict)
            losses.append(loss)
            sum_loss += loss
        return losses, sum_loss / len(predictions)
    

    def loss_weights_general(self, predictions, real, weights, w, indexes,loss_f, forward_func):
        losses = []
        sum_loss = 0
        samples=1
        if forward_func==self.forward:
            for i, predict in enumerate(predictions):
                loss_mean=loss_f(real, predict).mean(1)
                loss = (loss_mean * w[indexes]*samples).mean()
                losses.append(loss)
                sum_loss += loss
        else:
            for i, predict in enumerate(predictions):
                loss_mean=loss_f(real[weights], predict).mean(1)
                loss = (loss_mean * w[indexes]*samples).mean()
                losses.append(loss)
                sum_loss += loss
        return losses, sum_loss / len(predictions)

    def loss_weights(self, predictions, real, where_to_weight,w,indexes, loss_f):
        return self.loss_weights_general(predictions, real, where_to_weight,w,indexes,loss_f, self.forward)

    def loss_weights_back(self, predictions, real, where_to_weight,w, indexes,loss_f):
        return self.loss_weights_general(predictions, real, where_to_weight,w,indexes, loss_f, self.train_back_forward)

    def to_weights(self,log_w, temperature=10): 
        logsoft = nn.LogSoftmax(dim = 0)
        weights = torch.exp(logsoft(log_w/temperature))
        return weights
    
    def kde_density(self,X):
        is_cuda = X.is_cuda
        ldev = X.device  
        dim = X.size()[1] 
        kde = KernelDensity(kernel='gaussian', bandwidth=0.05, atol=0.2).fit(X.cpu().data.numpy())
        log_density = kde.score_samples(X.cpu().data.numpy())
        return log_density, kde
    
    def calc_EIs_kde(self,s,sp,samples,MSE,MSE_raw,L,bigL,weights_id,device,use_cuda='True'):
        encodes = self.encoding1(sp, weights_id)
        predicts1, latent1s, latentp1s = self.train_forward(s, weights_id)
        losses_returned_test, loss_test = self.loss(predicts1, sp, MSE)
        eis = []
        sigmass = []
        weightss = []
        dynamics = self.dynamics_modules[weights_id]
        #out the latent space
        latent1=latent1s[weights_id]
        latentp1=latentp1s[0]
        #The scale
        scale=len(latent1[0]) 
        encode=encodes[weights_id]
        latent1=latent1.cpu().detach().numpy()
        if latent1.shape[1]>100:
            scaler = StandardScaler()
            scaler.fit(latent1)
            latent1_zscore=scaler.transform(latent1)
            target_dim=10
            pca = PCA(n_components=target_dim)
            latent1_zscore= pca.fit_transform(latent1_zscore)
            latent1_zscore=torch.tensor(latent1_zscore)
            log_density, k_model_n = self.kde_density(latent1_zscore)
            log_density=torch.tensor(log_density,device=device)
        else:
            latent1=torch.tensor(latent1)
            log_density, k_model_n = self.kde_density(latent1)
            log_density=torch.tensor(log_density,device=device)
        log_rho = - scale * torch.log(2.0*torch.from_numpy(np.array(L)))  
        logp = log_rho - log_density  
        weights = self.to_weights(logp, temperature=1) * samples
        if use_cuda:
            weights = weights.cuda(device=device)
        weights=weights.unsqueeze(1)
        mse1 = MSE_raw(latentp1, encode)*weights
        sigmas=mse1.mean(axis=0)
        sigmas_matrix = torch.diag(sigmas)
        ei = approx_ei(scale, scale, sigmas_matrix.data, lambda x:(dynamics(x.unsqueeze(0))+x.unsqueeze(0)), 
                    num_samples = 1000, L=L, easy=True, device=device)  
        eis.append(ei)
        sigmass.append(sigmas)
        weightss.append(weights)
        return eis, sigmass,weightss, loss_test

    def multi_step_prediction(self, x, steps, weights):
        ss = self.encoding1(x, weights)
        i = len(ss)-1
        s = ss[i]
        temp_s_next_list = []
        for t in range(steps):
            s_next = self.dynamics_modules[i](s) + s
            if self.normalized_state:
                s_next = torch.tanh(s_next)
            temp_s_next_list.append(s_next)
            s = s_next
        y_list = []
        for j in range(len(temp_s_next_list)):
            y = self.decoding(temp_s_next_list[j], i)
            y_list.append(y)
        return y_list

def train_and_memorize(train_input,train_target,test_input,test_target,weights_id,folder_name,mice_id,stage,method,device,epoches=5e6+1,hidden_units = 64,scale=1,batch_size =100,train_stage='stage2',version=0):
    if version == 0:
        EIs=[]
        CEs=[]
        LOSSes, Loss_test=[], []
        term1_list, term2_list = [], []
        L=1
        cut_size=2
        epoches=int(epoches)
        hidden_units = hidden_units
        scale = scale
        batch_size = batch_size
        MAE = torch.nn.L1Loss()
        net = Parellel_Renorm_Dynamic(sym_size = train_input.shape[1], latent_size = scale, effect_size = train_input.shape[1],
                             cut_size=cut_size, hidden_units = hidden_units, normalized_state = True, device=device)
        net = net.to(device)               
        time_delay = 1 
        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad==True], lr=1e-4)
        for t in range(epoches):
            state_indexes = torch.multinomial(torch.ones(train_input.shape[0], device=device), num_samples=batch_size, replacement=False) 
            state = train_input[state_indexes]
            state_next = train_target[state_indexes]
            state = train_input
            state_next = train_target
            state = state.to(device)
            state_next = state_next.to(device)
            predicts,latents, latent_ps = net.forward(state,delay=time_delay)
            losses_returned,loss_train = net.loss(predicts, state_next, MAE)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            if t % 1000 == 0 and t >0:
                if test_input.shape[0] < 500:
                    state_for_EI = test_input
                    state_next_for_EI = test_target
                else:
                    test_state_indexes = torch.multinomial(torch.ones(test_input.shape[0], device=device), num_samples=800, replacement=False) 
                    state_for_EI=test_input[test_state_indexes]
                    state_next_for_EI=test_target[test_state_indexes]
                state_for_EI = state_for_EI.to(device)
                state_next_for_EI = state_next_for_EI.to(device)
                predicts,latents,latent_ps=net.forward(state_for_EI,delay=time_delay)
                losses_returned_test, loss_test = net.loss(predicts, state_next_for_EI, MAE)
                LOSSes.append([temp_loss.item() for temp_loss in losses_returned])
                Loss_test.append([temp_loss.item() for temp_loss in losses_returned_test])
                directory = f'./loc_result_stage1/{folder_name}/{mice_id}/{stage}'
                loss_array = np.array(LOSSes)
                loss_array_test = np.array(Loss_test)
                pd.DataFrame(loss_array).to_csv(os.path.join(directory, 'train_loss.csv'),index=None,header=None)
                pd.DataFrame(loss_array_test).to_csv(os.path.join(directory, 'test_loss.csv'),index=None,header=None)
                torch.save(net.state_dict(), f'./loc_model_stage1/{folder_name}/{mice_id}/{stage}.pkl')

    elif version==1:
        EIs=[]
        CEs=[]
        temp_ei, temp_model = 0, None
        LOSSes, LOSS_test=[], []        
        term1_list, term2_list = [], []
        L=1
        epoches=int(epoches)
        hidden_units = hidden_units
        scale = scale
        cut_size=2
        kernel = 'gaussian'
        bandwidth = 0.05
        atol = 0.2
        algorithm='kd_tree'
        batch_size =batch_size
        MAE = torch.nn.L1Loss()
        MAE_Raw = torch.nn.L1Loss(reduction='none')
        MSE_Raw = torch.nn.MSELoss(reduction='none')
        net = Parellel_Renorm_Dynamic(sym_size = train_input.shape[1], latent_size = scale, effect_size = train_input.shape[1],
                             cut_size=2, hidden_units = hidden_units, normalized_state = True, device=device)
        net = net.cuda(device)
        net.load_state_dict(torch.load(f'./loc_model_stage1/{folder_name}/{mice_id}/{stage}.pkl'))
        
        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad==True], lr=1e-4, weight_decay=1e-5)
        w_forward=torch.ones(train_input.shape[0],device=device)
        w_backward=torch.ones(train_input.shape[0],device=device)
        time_delay = 1
        for t in range(epoches):
            state_indexes = torch.multinomial(torch.ones(train_input.shape[0], device=device), num_samples=batch_size, replacement=False) 
            state=train_input[state_indexes]
            state_next=train_target[state_indexes]
            state = state.to(device)
            state_next = state_next.to(device)
            predicts,latents,latent_ps=net.train_forward(state,weights_id,delay=time_delay) 
            predicts_0,latents_0,latent_ps0=net.train_back_forward(state_next,weights_id,delay=time_delay)
            if t % 1000 == 0 and t!=(epoches-1) and t!=0:
                predicts_single, latents1, latent_ps1 = net.train_forward(train_input.to(device), weights_id)
                latents_kde = latents1[weights_id].cpu().data.numpy()
                if latents_kde.shape[1]>100:
                    scaler = StandardScaler()
                    scaler.fit(latents_kde)
                    latents_kde_1=scaler.transform(latents_kde)
                    target_dim=100
                    pca = PCA(n_components=target_dim)
                    latents_kde_1 = pca.fit_transform(latents_kde_1)
                else:
                    latents_kde_1=latents_kde
                kde = KernelDensity(kernel=kernel, bandwidth=bandwidth, atol=atol,algorithm=algorithm).fit(latents_kde_1)
                log_density=torch.tensor(kde.score_samples(latents_kde_1),device=device)
                scale=len(latents1[weights_id][0])
                log_rho = - scale * torch.log(2.0*torch.from_numpy(np.array(L)))  #均匀分布的概率分布
                logp = log_rho - log_density  
                w_backward = net.to_weights(logp, temperature=1)*train_input.shape[0]
            losses_returned,loss1 = net.loss_weights(predicts, state_next,weights_id,w_forward,state_indexes,MAE_Raw)
            losses_returned_0,loss2 = net.loss_weights_back(latent_ps0,latents,weights_id,w_backward,state_indexes, MAE_Raw)
            loss=losses_returned[0]+1*losses_returned_0[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if t % 2000 == 0:
                if test_input.shape[0] < 500:
                    state_for_EI = test_input
                    state_next_for_EI = test_target
                else:
                    test_state_indexes = torch.multinomial(torch.ones(test_input.shape[0], device=device), num_samples=800, replacement=False) 
                    state_for_EI=test_input[test_state_indexes]
                    state_next_for_EI=test_target[test_state_indexes]

                state_for_EI = state_for_EI.to(device)
                state_next_for_EI = state_next_for_EI.to(device)
                eis_kde,sigmass,weights,loss_test = net.calc_EIs_kde(state_for_EI,state_next_for_EI,state_for_EI.shape[0],MAE,MSE_Raw,L,L,weights_id,device)
                EIs.append([ei[0] for ei in eis_kde])
                term1_list.append([ei[3] for ei in eis_kde])
                term2_list.append([ei[4] for ei in eis_kde])
                LOSSes.append(loss1.item())
                LOSS_test.append(loss_test.item())
                directory = f'./loc_result_stage2/{folder_name}/{mice_id}/{stage}'
                loss_array = np.array(LOSSes)
                loss_array_test = np.array(LOSS_test)
                pd.DataFrame(loss_array).to_csv(os.path.join(directory, 'train_loss%s.csv'%weights_id),index=None,header=None)
                pd.DataFrame(loss_array_test).to_csv(os.path.join(directory, 'test_loss%s.csv'%weights_id),index=None,header=None)
                
                term1_list_array = np.array(term1_list)
                pd.DataFrame(term1_list_array).to_csv(os.path.join(directory, 'term1_scale%s.csv'%weights_id),index=None,header=None)

                term2_list_array = np.array(term2_list)
                pd.DataFrame(term2_list_array).to_csv(os.path.join(directory, 'term2_scale%s.csv'%weights_id),index=None,header=None)
                EI_array = np.array(EIs)
                pd.DataFrame(EI_array).to_csv(os.path.join(directory, 'EI_scale%s.csv'%weights_id),index=None,header=None)
                torch.save(net.state_dict(), f'./loc_model_stage2/{folder_name}/{mice_id}/{stage}_scale%s.pkl'%weights_id)

