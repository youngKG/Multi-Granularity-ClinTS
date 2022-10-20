from math import inf
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pickle
import sys
import os
from sklearn import model_selection
import pandas as pd
from tqdm import tqdm

Constants_PAD = 0


def proc_hii_data(x, y, input_dim, args):
    x = x[:, :input_dim*2+1]
    if args.task == "los":
        y = y-1

    x = np.transpose(x, (0, 2, 1))
    new_x = np.empty((len(x), len(x[1]), input_dim*3+1))
    print("data preprocessing in batch...")
    total = len(x)
    batch_sz = 5000

    pbar = tqdm(range(0, total, batch_sz)) 
    for start in pbar:
        end = min(start+batch_sz, total)
        
        new_x[start:end, :, :input_dim*2+1] = process_data(x[start:end], input_dim)
        new_x[start:end, :, input_dim*2+1:input_dim*3+1] = cal_tau(x[start:end, :, -1], x[start:end, :, input_dim:2 * input_dim])

    print("data preprocess in batch done.")
    return new_x, y


def get_clints_hii_data(args):
    
    if args.task == "vent" or args.task == "vaso":
        data_folder_x = args.root_path + args.data_path + 'cip/' 
        data_folder_y = args.root_path + args.data_path + 'cip/' + args.task + '_'
    else:
        data_folder_x = args.root_path + args.data_path + args.task + '/' 
        data_folder_y = args.root_path + args.data_path + args.task + '/' 
    
    dataloader = []

    for set_name in ['train', 'val', 'test']:
        data_x_all = []
        data_y_all = []
        
        if set_name == 'train':
            shuffle = True
        else:
            shuffle = False
            
        print("loading " + set_name + " data")

        ######## if OOM kill occur in loading data, try to split training set into 5 part and load data in batch
        if set_name == "train" and args.task != "mor" and args.load_in_batch:
            for i in range(5):
                data_x = np.load(data_folder_x + set_name + '_input' + str(i) + '.npy', allow_pickle=True)
                data_y = np.load(data_folder_y + set_name + '_output' + str(i) + '.npy', allow_pickle=True)
                
                args.num_types = int((data_x.shape[1] - 1) / 2)
                data_x, data_y = proc_hii_data(data_x, data_y, args.num_types, args)
                data_x_all.append(data_x)
                data_y_all.append(data_y)
                del data_x, data_y
            
            data_x_all = np.concatenate(data_x_all)
            data_y_all = np.concatenate(data_y_all)
            
        else:
            data_y_all = np.load(data_folder_y + set_name + '_output.npy', allow_pickle=True)
            data_x_all = np.load(data_folder_x + set_name + '_input.npy', allow_pickle=True)
            
        args.num_types = int((data_x_all.shape[1] - 1) / 2)
        data_x_all, data_y_all = proc_hii_data(data_x_all, data_y_all, args.num_types, args)

        print(data_x_all.shape, data_y_all.shape)
        dataloader.append(get_data_loader(data_x_all, data_y_all, args, shuffle=shuffle))
        del data_x_all, data_y_all
        
    print("type num: ", args.num_types)
    return dataloader[0], dataloader[1], dataloader[2]


def get_data_loader(data_x, data_y, args, shuffle=False):
    data_combined = TensorDataset(torch.from_numpy(data_x).float(),
                                        torch.from_numpy(data_y).long().squeeze())
    dataloader = DataLoader(
        data_combined, batch_size=args.batch_size, shuffle=shuffle)
    
    return dataloader

def cal_tau(observed_tp, observed_mask):
    if observed_tp.ndim == 2:
        tmp_time = observed_mask * np.expand_dims(observed_tp,axis=-1) # [B,L,K]
    else:
        tmp_time = observed_tp.copy()
        
    b,l,k = tmp_time.shape
    
    new_mask = observed_mask.copy()
    new_mask[:,0,:] = 1
    tmp_time[new_mask == 0] = np.nan
    tmp_time = tmp_time.transpose((1,0,2)) # [L,B,K]
    tmp_time = np.reshape(tmp_time, (l,b*k)) # [L, B*K]

    # padding the missing value with the next value
    df1 = pd.DataFrame(tmp_time)
    df1 = df1.fillna(method='ffill')
    tmp_time = np.array(df1)

    tmp_time = np.reshape(tmp_time, (l,b,k))
    tmp_time = tmp_time.transpose((1,0,2)) # [B,L,K]
    
    tmp_time[:,1:] -= tmp_time[:,:-1]
    del new_mask
    return tmp_time * observed_mask

def process_data(x, input_dim ):
    observed_vals, observed_mask, observed_tp = x[:, :,
                                                :input_dim], x[:, :, input_dim:2 * input_dim], x[:, :, -1]

    observed_tp = np.expand_dims(observed_tp, axis=-1)
    observed_vals = tensorize_normalize(observed_vals)
    observed_vals[observed_mask == 0] = 0
    x = np.concatenate((observed_vals, observed_mask, observed_tp), axis=-1)
    return x


def tensorize_normalize(P_tensor):
    mf, stdf = getStats(P_tensor)
    P_tensor = normalize(P_tensor, mf, stdf)
    return P_tensor

def getStats(P_tensor):
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    mf = np.zeros((F, 1))
    stdf = np.ones((F, 1))
    eps = 1e-7
    for f in range(F):
        vals_f = Pf[f, :]
        vals_f = vals_f[vals_f > 0]
        if len(vals_f) > 0:
            mf[f] = np.mean(vals_f)
            tmp_std = np.std(vals_f)
            stdf[f] = np.max([tmp_std, eps])
    return mf, stdf

def normalize(P_tensor, mf, stdf):
    """ Normalize time series variables. Missing ones are set to zero after normalization. """
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    for f in range(F):
        Pf[f] = (Pf[f]-mf[f])/(stdf[f]+1e-18)
    Pnorm_tensor = Pf.reshape((F, N, T)).transpose((1, 2, 0))
    return Pnorm_tensor

