import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
# from utils.timefeatures import time_features
# from data_provider.m4 import M4Dataset, M4Meta
# from data_provider.uea import subsample, interpolate_missing, Normalizer
from data_provider.patch_mask import *
# from data_provider.test import *
# from sktime.utils import load_data
import warnings
import torch.nn as nn
warnings.filterwarnings('ignore')

class PSMSegLoader(Dataset):
    def __init__(self, data, train_labels, test_data, test_labels, root_path, win_size,  step=1, flag="train", test_flag = 0, 
                patch_len=10, patch_stride=10,mask_ratio=0.2, c_out=25,seleck_k=3,periods_num=5,multi_resolution=1):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.val_labels = self.train[(int)(data_len * 0.8):]
        self.test_labels = test_labels
        self.win_size = win_size
        self.patch_len = patch_len
        self.stride = patch_stride
        self.mask_ratio = mask_ratio
        self.train_labels = train_labels
        self.test_flag = test_flag
        self.c_out = c_out
        self.seleck_k = seleck_k
        self.periods = periods_num
        self.multi_resolution = multi_resolution

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            sequence_window,label_window = np.float32(self.train[index:index + self.win_size]), np.float32(self.train_labels[0:self.win_size])
        elif (self.flag == 'val'):
            sequence_window,label_window = np.float32(self.val[index:index + self.win_size]), np.float32(self.val_labels[0:self.win_size])
        elif (self.flag == 'test'):
            sequence_window,label_window = np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            sequence_window,label_window = np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        
        if (self.flag == 'test'): 
            if (self.multi_resolution==1):
                xb_patch, yb_patch, num_patch = create_patch(sequence_window, label_window, 10, 10)
           
            num_patch, nvars, patch_length = xb_patch.shape
            xb_patch_permuted = xb_patch.permute(0, 2, 1)
            xb_patch = xb_patch_permuted.reshape(-1, nvars)
            yb_patch = yb_patch.view(-1)
            return xb_patch,yb_patch,0
        if (self.multi_resolution==1):
            top_k_frequencies, top_k_periods = extract_top_k_periods(sequence_window,periods = self.periods)
            all_patch = []
            
            
            for patch_len in top_k_periods:
                xb_patch, yb_patch, num_patch = create_patch(sequence_window, label_window, int(patch_len), int(patch_len))
                xb_mask, _, self.mask, _ = random_masking(xb_patch, self.mask_ratio) 
                num_patch, nvars, patch_length = xb_mask.shape
                xb_mask_permuted = xb_mask.permute(0, 2, 1)
                xb_mask = xb_mask_permuted.reshape(-1, nvars)
                all_patch.append(xb_mask) #torch.Size([100, 38])
            
            all_xb_patch = select_representative_vectors(all_patch,k=self.seleck_k)

            self.fc = nn.Linear(self.c_out, self.c_out)
            all_x_enc = []
            for orig_x_enc in all_xb_patch:
                all_x_enc.append(self.fc(orig_x_enc).detach().numpy())
                

            pearson_diff_max_min = pearson_correlation_matrix(all_x_enc)
            weight_x = calculate_weights(pearson_diff_max_min)
            xb_patch = np.zeros_like(all_x_enc[0]) #torch.zeros_like(all_x_enc[0]) 

            for i, x_enc_1 in enumerate(all_x_enc):
                xb_patch += weight_x[i] * x_enc_1  
           
            yb_patch = yb_patch.permute(0, 2, 1)
            yb_patch = yb_patch.reshape(-1, yb_patch.shape[-1])
            return xb_patch, yb_patch, 0


    

class MSLSegLoader(Dataset):
    def __init__(self, data, train_labels, test_data, test_labels, root_path, win_size,  step=1, flag="train", test_flag = 0, 
                patch_len=10, patch_stride=10,mask_ratio=0.2,c_out=55,seleck_k=3,periods_num=5,multi_resolution=1):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.val_labels = self.train[(int)(data_len * 0.8):]
        self.test_labels = test_labels
        self.win_size = win_size
        self.patch_len = patch_len
        self.stride = patch_stride
        
        self.mask_ratio = mask_ratio
        self.train_labels = train_labels
        self.test_flag = test_flag
        self.c_out = c_out
        self.seleck_k = seleck_k
        self.periods = periods_num
        self.multi_resolution = multi_resolution

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            sequence_window,label_window = np.float32(self.train[index:index + self.win_size]), np.float32(self.train_labels[0:self.win_size])
        elif (self.flag == 'val'):
            sequence_window,label_window = np.float32(self.val[index:index + self.win_size]), np.float32(self.val_labels[0:self.win_size])
        elif (self.flag == 'test'):
            sequence_window,label_window = np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            sequence_window,label_window = np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        
        if (self.flag == 'test'): 
            if (self.multi_resolution==1):
                xb_patch, yb_patch, num_patch = create_patch(sequence_window, label_window, 10, 10)
           
            num_patch, nvars, patch_length = xb_patch.shape
            xb_patch_permuted = xb_patch.permute(0, 2, 1)
            xb_patch = xb_patch_permuted.reshape(-1, nvars)
            yb_patch = yb_patch.view(-1)
            return xb_patch,yb_patch,0
        if (self.multi_resolution==1):
            top_k_frequencies, top_k_periods = extract_top_k_periods(sequence_window,periods = self.periods)
            all_patch = []
            
            
            for patch_len in top_k_periods:
                xb_patch, yb_patch, num_patch = create_patch(sequence_window, label_window, int(patch_len), int(patch_len))
                xb_mask, _, self.mask, _ = random_masking(xb_patch, self.mask_ratio) 
                num_patch, nvars, patch_length = xb_mask.shape
                xb_mask_permuted = xb_mask.permute(0, 2, 1)
                xb_mask = xb_mask_permuted.reshape(-1, nvars)
                all_patch.append(xb_mask) #torch.Size([100, 38])
            
            all_xb_patch = select_representative_vectors(all_patch,k=self.seleck_k)

            self.fc = nn.Linear(self.c_out, self.c_out)
            all_x_enc = []
            for orig_x_enc in all_xb_patch:
                all_x_enc.append(self.fc(orig_x_enc).detach().numpy())
                

            pearson_diff_max_min = pearson_correlation_matrix(all_x_enc)
            weight_x = calculate_weights(pearson_diff_max_min)
            xb_patch = np.zeros_like(all_x_enc[0]) #torch.zeros_like(all_x_enc[0]) 

            for i, x_enc_1 in enumerate(all_x_enc):
                xb_patch += weight_x[i] * x_enc_1  
           
            yb_patch = yb_patch.permute(0, 2, 1)
            yb_patch = yb_patch.reshape(-1, yb_patch.shape[-1])
            return xb_patch, yb_patch, 0


class SMDSegLoader(Dataset):
    def __init__(self, data, train_labels, test_data, test_labels, root_path, win_size,  step=1, flag="train", test_flag = 0,
                patch_len=10, patch_stride=10,mask_ratio=0.3,c_out=38,seleck_k=3,periods_num=5,multi_resolution=1):
        self.flag = flag
        self.step = step 
        self.win_size = win_size
        self.patch_len = patch_len
        self.stride = patch_stride
        self.mask_ratio = mask_ratio
        self.scaler = StandardScaler()
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.val_labels = self.train[(int)(data_len * 0.8):]
        self.test_labels = test_labels
        self.train_labels = train_labels
        self.test_flag = test_flag
        self.c_out = c_out
        self.seleck_k = seleck_k
        self.periods = periods_num
        self.multi_resolution = multi_resolution

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        # print("get_items,index:",index)
        
        if self.flag == "train":
            sequence_window,label_window = np.float32(self.train[index:index + self.win_size]), np.float32(self.train_labels[0:self.win_size])            
        elif (self.flag == 'val'):
            sequence_window,label_window  = np.float32(self.val[index:index + self.win_size]), np.float32(self.val_labels[0:self.win_size])
        elif (self.flag == 'test'):
            sequence_window,label_window  = np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            sequence_window,label_window  = np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        if (self.flag == 'test'): 
            if (self.multi_resolution==1):
                xb_patch, yb_patch, num_patch = create_patch(sequence_window, label_window, 10, 10)
           
            num_patch, nvars, patch_length = xb_patch.shape
            xb_patch_permuted = xb_patch.permute(0, 2, 1)
            xb_patch = xb_patch_permuted.reshape(-1, nvars)
            yb_patch = yb_patch.view(-1)
            return xb_patch,yb_patch,0
        if (self.multi_resolution==1):
            top_k_frequencies, top_k_periods = extract_top_k_periods(sequence_window,periods = self.periods)
            all_patch = []
            
            
            for patch_len in top_k_periods:
                xb_patch, yb_patch, num_patch = create_patch(sequence_window, label_window, int(patch_len), int(patch_len))
                xb_mask, _, self.mask, _ = random_masking(xb_patch, self.mask_ratio) 
                num_patch, nvars, patch_length = xb_mask.shape
                xb_mask_permuted = xb_mask.permute(0, 2, 1)
                xb_mask = xb_mask_permuted.reshape(-1, nvars)
                all_patch.append(xb_mask) #torch.Size([100, 38])
            
            all_xb_patch = select_representative_vectors(all_patch,k=self.seleck_k)

            self.fc = nn.Linear(self.c_out, self.c_out)
            all_x_enc = []
            for orig_x_enc in all_xb_patch:
                all_x_enc.append(self.fc(orig_x_enc).detach().numpy())
                

            pearson_diff_max_min = pearson_correlation_matrix(all_x_enc)
            weight_x = calculate_weights(pearson_diff_max_min)
            xb_patch = np.zeros_like(all_x_enc[0]) #torch.zeros_like(all_x_enc[0]) 

            for i, x_enc_1 in enumerate(all_x_enc):
                xb_patch += weight_x[i] * x_enc_1  
           
            yb_patch = yb_patch.permute(0, 2, 1)
            yb_patch = yb_patch.reshape(-1, yb_patch.shape[-1])
            return xb_patch, yb_patch, 0

    


class SWATSegLoader(Dataset):
    def __init__(self, data, train_labels, test_data, test_labels, root_path, win_size,  step=1, flag="train", test_flag = 0, 
    patch_len=10, patch_stride=10,mask_ratio=0.2,c_out=51,seleck_k=3,periods_num=5,multi_resolution=1):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        train_data = data
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.val_labels = self.train[(int)(data_len * 0.8):]
        self.test = test_data
        self.test_labels = test_labels

        self.patch_len = patch_len
        self.stride = patch_stride
        self.mask_ratio = mask_ratio
        self.train_labels = train_labels
        self.test_flag = test_flag
        self.c_out = c_out
        self.seleck_k = seleck_k
        self.periods = periods_num
        self.multi_resolution = multi_resolution

    def __len__(self):
        if self.flag == "train":
            tmp = (self.train.shape[0] - self.win_size) // self.step + 1
            return tmp
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            sequence_window,label_window = np.float32(self.train[index:index + self.win_size]), np.float32(self.train_labels[0:self.win_size])
        elif (self.flag == 'val'):
            sequence_window,label_window = np.float32(self.val[index:index + self.win_size]), np.float32(self.val_labels[0:self.win_size])
        elif (self.flag == 'test'):
            sequence_window,label_window = np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            sequence_window,label_window = np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        
        if (self.flag == 'test'): 
            if (self.multi_resolution==1):
                xb_patch, yb_patch, num_patch = create_patch(sequence_window, label_window, 10, 10)
           
            num_patch, nvars, patch_length = xb_patch.shape
            xb_patch_permuted = xb_patch.permute(0, 2, 1)
            xb_patch = xb_patch_permuted.reshape(-1, nvars)
            yb_patch = yb_patch.view(-1)
            return xb_patch,yb_patch,0
        if (self.multi_resolution==1):
            top_k_frequencies, top_k_periods = extract_top_k_periods(sequence_window,periods = self.periods)
            all_patch = []
            
            
            for patch_len in top_k_periods:
                xb_patch, yb_patch, num_patch = create_patch(sequence_window, label_window, int(patch_len), int(patch_len))
                xb_mask, _, self.mask, _ = random_masking(xb_patch, self.mask_ratio) 
                num_patch, nvars, patch_length = xb_mask.shape
                xb_mask_permuted = xb_mask.permute(0, 2, 1)
                xb_mask = xb_mask_permuted.reshape(-1, nvars)
                all_patch.append(xb_mask) #torch.Size([100, 38])
            
            all_xb_patch = select_representative_vectors(all_patch,k=self.seleck_k)

            self.fc = nn.Linear(self.c_out, self.c_out)
            all_x_enc = []
            for orig_x_enc in all_xb_patch:
                all_x_enc.append(self.fc(orig_x_enc).detach().numpy())
                

            pearson_diff_max_min = pearson_correlation_matrix(all_x_enc)
            weight_x = calculate_weights(pearson_diff_max_min)
            xb_patch = np.zeros_like(all_x_enc[0]) #torch.zeros_like(all_x_enc[0]) 

            for i, x_enc_1 in enumerate(all_x_enc):
                xb_patch += weight_x[i] * x_enc_1  
           
            yb_patch = yb_patch.permute(0, 2, 1)
            yb_patch = yb_patch.reshape(-1, yb_patch.shape[-1])
            return xb_patch, yb_patch, 0



class UCRSegLoader(Dataset):
    def __init__(self, data, train_labels, test_data, test_labels, root_path, win_size,  step=1, flag="train", test_flag = 0,
    patch_len=10, patch_stride=10,mask_ratio=0.2,c_out=1,seleck_k=3,periods_num=5,multi_resolution=1):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = test_labels
        self.win_size = win_size
        self.patch_len = patch_len
        self.stride = patch_stride
        self.mask_ratio = mask_ratio
        self.train_labels = train_labels
        self.test_flag = test_flag
        self.c_out = c_out
        self.seleck_k = seleck_k
        self.periods = periods_num
        self.multi_resolution = multi_resolution

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            sequence_window,label_window = np.float32(self.train[index:index + self.win_size]), np.float32(self.train_labels[0:self.win_size])
        elif (self.flag == 'val'):
            sequence_window,label_window = np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            sequence_window,label_window = np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            sequence_window,label_window = np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        if (self.flag == 'test'): 
            if (self.multi_resolution==1):
                xb_patch, yb_patch, num_patch = create_patch(sequence_window, label_window, 10, 10)
           
            num_patch, nvars, patch_length = xb_patch.shape
            xb_patch_permuted = xb_patch.permute(0, 2, 1)
            xb_patch = xb_patch_permuted.reshape(-1, nvars)
            yb_patch = yb_patch.view(-1)
            return xb_patch,yb_patch,0
        if (self.multi_resolution==1):
            top_k_frequencies, top_k_periods = extract_top_k_periods(sequence_window,periods = self.periods)
            all_patch = []
            
            
            for patch_len in top_k_periods:
                xb_patch, yb_patch, num_patch = create_patch(sequence_window, label_window, int(patch_len), int(patch_len))
                xb_mask, _, self.mask, _ = random_masking(xb_patch, self.mask_ratio) 
                num_patch, nvars, patch_length = xb_mask.shape
                xb_mask_permuted = xb_mask.permute(0, 2, 1)
                xb_mask = xb_mask_permuted.reshape(-1, nvars)
                all_patch.append(xb_mask) #torch.Size([100, 38])
            
            all_xb_patch = select_representative_vectors(all_patch,k=self.seleck_k)

            self.fc = nn.Linear(self.c_out, self.c_out)
            all_x_enc = []
            for orig_x_enc in all_xb_patch:
                all_x_enc.append(self.fc(orig_x_enc).detach().numpy())
                

            pearson_diff_max_min = pearson_correlation_matrix(all_x_enc)
            weight_x = calculate_weights(pearson_diff_max_min)
            xb_patch = np.zeros_like(all_x_enc[0]) #torch.zeros_like(all_x_enc[0]) 

            for i, x_enc_1 in enumerate(all_x_enc):
                xb_patch += weight_x[i] * x_enc_1  
           
            yb_patch = yb_patch.permute(0, 2, 1)
            yb_patch = yb_patch.reshape(-1, yb_patch.shape[-1])
            return xb_patch, yb_patch, 0


       
