from data_provider.data_factory import data_provider

from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
# import torch.multiprocessing
import torch.multiprocessing as mp
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import csv
from torchstat import stat
mp.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
from torch.utils import data as data_utils
import os
import gc
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import random
from sklearn.neighbors import KernelDensity
from sklearn.metrics import mutual_info_score

warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from thop import profile

import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed
import ast
from pate.PATE_metric import PATE


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)
        self.mask_matrix_arg = ast.literal_eval(self.args.mask_matrix_arg)

    def _build_model(self):
        self.model = {}
        datasets = ast.literal_eval(self.args.dataset_names)
        
        for dataset in datasets:
            self.model[dataset] = self.model_dict['GPT4TS'].Model(dataset, self.args).float()
        # self.server_model = self.model['SMD']
        return self.model

    def _get_data(self, flag,shared=False):
        data_set, data_loader = data_provider(self.args, flag,shared)
        # num_clients = len(list(data_loader.keys()))
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = {}
        self.vae_local_optimizer = None
        return model_optim,self.vae_local_optimizer

    def _select_criterion(self, classificate=False):
        if classificate == False:
            criterion = nn.MSELoss()

        else:
            criterion = nn.BCELoss()

        return criterion
    def check_valid_names(self, selected_keys, valid_name):
        matches = {}
        for key in selected_keys:
            if valid_name.startswith(key):
                return True
        return False
    def get_trainable_params(self, model, mask_matrix, threshold = 0.5):

        trainable_params = {}
        for i, layer in enumerate(model):
            for j, (name, param) in enumerate(layer.named_parameters()):
                if param.requires_grad:
                    # if 'wpe' in name or 'mlp' in name:
                    #     trainable_params[str(i)+'.'+name] = param
                    # else:
                    valid_name = str(i)+'_' +name.replace('.', '_')
                    if self.check_valid_names(mask_matrix, valid_name):
                        trainable_params[name] = param
        return trainable_params

    def frobenius_distance_loss(self,w_agg, w_clients, device):
       
        w_agg = w_agg.to(w_clients.device)
        torch.cuda.empty_cache()
        distances = [torch.norm(w - w_agg, p='fro') for w in w_clients]
        avg_distance = torch.mean(torch.stack(distances)) 
        return avg_distance
    def aggregate_weights_w1(self, w1_dict, data_nums):
        
        device = list(w1_dict.values())[0].device  
        w_agg = torch.zeros_like(list(w1_dict.values())[0]).to(device)  

        total_data = sum(data_nums.values()) 

        for client_id, w1 in w1_dict.items():
            weight = data_nums[client_id] / total_data
            w_agg += weight * w1.to(device)  

        return w_agg
    def aggregate_weights_w1(self, w1_dict, data_nums, device='cuda:3'):
        total_data = sum(data_nums.values())  
        w_agg = None  

        for client_id, w1 in w1_dict.items():
            weight = data_nums[client_id] / total_data  
            w1 = w1.to(device)  
            weight = torch.tensor(weight, device=device) 
            if w_agg is None:
                w_agg = weight * w1  
            else:
                w_agg += weight * w1  
        return w_agg

    def aggregate_client(self, client_params_list, client_data_sizes, device='cuda:3'):

        param_sums = {}
        param_total_data_size = {}
        self.global_params = {}

        for (dataset, client_params) in client_params_list.items():
            for param_name, param_value in client_params.items():
                if param_name not in param_sums:
                    client_data_size = client_data_sizes[dataset]
                    param_sums[param_name] = client_data_size * param_value.clone()
                    param_total_data_size[param_name] = client_data_size
                else:
                    param_on_device = param_value.to(device)
                    param_sums[param_name] = param_sums[param_name].to(device)
                    param_sums[param_name] += client_data_size * param_on_device
                    param_total_data_size[param_name] += client_data_size

        for param_name in param_sums:
            self.global_params[param_name] = param_sums[param_name] / param_total_data_size[param_name]

        aggregated_client_params = {}
        for dataset, client_params in client_params_list.items():
            aggregated_params = {param_name: self.global_params[param_name] for param_name, param_value in client_params.items()}
            aggregated_client_params[dataset] = aggregated_params
        return aggregated_client_params, self.global_params

#---------------------------------------------------
    def federated_aggregation(self, model_dicts, data_nums, device='cuda:0'):
        total_data = sum(data_nums.values())
        data_ratio = {client_id: count / total_data for client_id, count in data_nums.items()}  

        global_params = {name: torch.zeros_like(param).to(device) for name, param in self.server_model.gpt2_main.named_parameters()} # if param.requires_grad
        # model_dicts = model_dicts.to(device)

        for state_dict, client_id in zip(model_dicts, list(data_ratio.keys())):
            for name, param in state_dict.items():
                if name in global_params:
                    param_on_device = param.to(device)
                    global_params[name] += param_on_device * data_ratio[client_id]

        return global_params

    def load_global_params(self, model, global_params):
        if global_params==None:
            return
        for name, param in model.named_parameters():
            if name in global_params and param.requires_grad:
                param.data.copy_(global_params[name])

    def load_global_params_server(self, model, global_params):
        if global_params==None:
            return
        for name, param in model.named_parameters():
            name = name
            if name in global_params and param.requires_grad:
                param.data.copy_(global_params[name])

    def vali(self, vali_data, vali_loader, criterion,dataset,flag):
        # print("======================VALI MODE======================")
        total_loss = []
        self.model[dataset].eval()
        with torch.no_grad():
            for i, (batch_x, _, mask_bool) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device[dataset])

                self.model[dataset].to(batch_x.device)
                client_A = self.model[dataset].forward(batch_x)
                
                client_device = client_A.device
                
                self.server_model = self.server_model.to(client_device)
                server_B = self.server_model.Forward2(self.server_model, client_A)
                outputs = self.model[dataset].Forward_3(self.model[dataset].out_layer,server_B,flag)
                f_dim = -1 if self.args.features == 'MS' else 0
                if not flag:
                    outputs = outputs[:, :, f_dim:]
                else:
                    outputs = outputs
                pred = outputs.detach().cpu()
                batch_x = batch_x.detach().cpu()
     
                loss = criterion(pred, batch_x)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model[dataset].train()
        return total_loss


    def train(self, setting):
        print("======================TRAIN MODE======================")
        self.model_optim = {}
        self.model_optim_B = {}
        self.model_optim_mask = {}
        self.optimizer_for_weight = {}
        self.global_params_A = None
        self.global_params = None

        torch.cuda.empty_cache()
        train_data, train_loader = self._get_data(flag='train') 
        self.num_clients = len(list(train_loader.keys()))

        train_list = [item for item in train_loader] 
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints,self.args.model_id + '_' + self.args.save_path)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader) 
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        data_nums = {} 
        model_dicts = [] 
        lambda_1 = 0.50
        trainable_params_A = []
        self.bce_list = ['UCR']

        # self.server_model = self.model_dict['GPT4TS'].Model('server', self.args).float().to(self.device)
        self.learning_rate = ast.literal_eval(self.args.learning_rate)
        w1_dict = {}
        w1, w2, error = None, None, None
        self.w1_agg = None
        for global_epoch in range(self.args.train_epochs):           
            model_dicts_A = {} 
            model_dicts_server = []
            client_gflops = {}

            print(f'\n | Global Training Round : {global_epoch + 1} |\n')
            

            for _client_id, data_loader in train_loader.items():
                dataset = _client_id[9:]
                print(dataset)
                if (dataset in self.bce_list):
                    flag = True
                else:
                    flag = False
                criterion = self._select_criterion(classificate=flag) 
                client_id = dataset
                
                data_nums[dataset] = len(data_loader) 
                
                # gpu_tracker.clear_cache()
                self.model[dataset].to(self.device[client_id])
                self.server_model = self.model[dataset]
               
                if global_epoch>0:
                    print('load aggregated gpt2_client params..')
                    self.load_global_params(self.model[dataset].gpt2_client, self.aggregated_client_params[dataset])
                    print('load aggregated server_model params..')
                    self.load_global_params_server(self.server_model.gpt2_main, self.global_params)
                

                self.model[dataset].train()
                self.model[dataset].out_layer.train()
                
                self.server_model.train()

                # self.load_global_params(self.server_model, self.global_params)

                self.model_optim[dataset] = optim.Adam(self.model[dataset].gpt2_client.parameters(), lr=self.learning_rate[dataset])
                self.model_optim_B[dataset] = optim.Adam(self.model[dataset].out_layer.parameters(), lr=self.learning_rate[dataset])
                optimizer_server = optim.Adam(self.server_model.gpt2_main.parameters(), lr=self.learning_rate[dataset])
                self.model_optim_mask[dataset] = optim.Adam(self.model[dataset].mask_matrix.parameters(), lr=self.learning_rate[dataset])
                self.optimizer_for_weight[dataset] = optim.Adam(self.model[dataset].gpt2_client[-1].attn.c_attn.parameters(), lr=1e-4)

                for epoch in range(self.args.local_epoch):
                    train_loss = []
                    for i, (batch_x, batch_y, mask_bool) in enumerate(data_loader):
                  
                        batch_x = batch_x.float().to(self.device[dataset])
                        batch_y = batch_y.float().to(self.device[dataset])
                      
                        
                        self.model_optim[dataset].zero_grad()
                        self.model_optim_mask[dataset].zero_grad()
                        self.optimizer_for_weight[dataset].zero_grad()
                        client_A = self.model[dataset].forward(batch_x)
                        
                        optimizer_server.zero_grad()
                        server_B = self.server_model.Forward2(self.server_model, client_A)

                        
                        self.model_optim_B[dataset].zero_grad()
                        dec_out = self.model[dataset].Forward_3(self.model[dataset].out_layer,server_B,flag)
                        param_loss = 0.0
                        
                        f_dim = -1 if self.args.features == 'MS' else 0
                        if not flag:
                            outputs = dec_out[:, :, f_dim:]
                        else:
                            outputs = dec_out
                        del dec_out
                        # torch.cuda.empty_cache()
                        if flag==False:
                            criterion_loss = criterion(outputs, batch_x)
                        else:
                            criterion_loss = criterion(outputs, batch_y)
                        # self.w1_agg = None    
                        if self.w1_agg != None:
                            # w1 = 0
                            w1, w2, error = self.model[dataset].decompose_last_layer_svd()
                            # param_loss =  self.frobenius_distance_loss(self.w1_agg, w1,self.device[client_id])
                            w1 = w1.to(self.w1_agg.device)
                            param_loss = torch.norm((self.w1_agg - w1)) ** 2
                           

                            param_loss = param_loss.to(criterion_loss.device)
                          
                        loss = criterion_loss + param_loss*self.args.param_loss_coef

                        train_loss.append(loss.item())
                        
                        if self.w1_agg != None:
                                
                            loss.backward(retain_graph=True) #
                        else:
                            loss.backward() #
                   
                        del batch_x, batch_y, mask_bool, loss, outputs, param_loss, criterion_loss
                        
                        server_dB = server_B.grad.clone().detach()
                        self.model_optim_B[dataset].step()
                       
                        client_dA = self.model[dataset].Backward2(server_B, server_dB, client_A, optimizer_server)

                       
                        self.selected_keys = self.model[dataset].Backward3(client_A, client_dA, self.model_optim[dataset], self.model_optim_mask[dataset],self.optimizer_for_weight[dataset]) #self.model[dataset].gpt2_client
                        del client_A, client_dA,server_B,server_dB
                        if i%100==0:
                            torch.cuda.empty_cache()
                        gc.collect()
                        
                        for param in self.model[dataset].parameters():
                            if param.grad is not None:
                                param.grad.detach_()
                                param.grad = None
                                del param.grad
                    
                   
                   
                    w1, w2, error = self.model[dataset].decompose_last_layer_svd()

                    client_train_loss = sum(train_loss) / len(train_loss) 
                
                    client_vali_loader = vali_loader[_client_id]
                    client_vali_data = vali_data[_client_id]       
                    vali_loss = self.vali(client_vali_data, client_vali_loader, criterion, client_id, flag) 
                
                    print("{0}, Local Epoch: {1}, Steps: {2} | Train Loss: {3:.7f} Vali Loss: {4:.7f}".format(
                    client_id, epoch + 1, train_steps, client_train_loss, vali_loss))  
                
                    torch.cuda.empty_cache()
                    gc.collect()
                  
                w1_dict[dataset] = w1
                print('self.selected_keys: ', self.selected_keys) 
                if dataset in self.mask_matrix_arg:
                    print("use mask_matrix_arg...")
                    trainable_params_A = self.get_trainable_params(self.model[dataset].gpt2_client, self.selected_keys) ##use mask matrix
                else:
                    trainable_params_A = {name: param for name, param in self.model[dataset].gpt2_client.named_parameters() if param.requires_grad}

                trainable_params_server = {name: param for name, param in self.server_model.gpt2_main.named_parameters() if param.requires_grad}

                
                model_dicts_A[dataset] = trainable_params_A
                model_dicts_server.append(trainable_params_server)
           
            self.aggregated_client_params, self.aggregated_global_params= self.aggregate_client(model_dicts_A,data_nums)
            self.w1_agg = self.aggregate_weights_w1(w1_dict,data_nums)

            # server(gpt_main)
            data_nums_server = {client_id: 1 for client_id in range(self.num_clients)}  
            self.global_params = self.federated_aggregation(model_dicts_server, data_nums_server)
            
            if global_epoch == self.args.train_epochs-1:
                print('new load aggregated server_model params..')
                self.load_global_params_server(self.server_model.gpt2_main, self.global_params)
                for dataset in self.aggregated_client_params.keys():
                    print('load aggregated gpt2_client params..')
                    self.load_global_params(self.model[dataset].gpt2_client, self.aggregated_client_params[dataset])
                
            print("Global Epoch: {}".format(global_epoch + 1))

                        
        return self.model

        


    def test(self, setting, test=0):  
        print("======================TEST MODE======================")

        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        folder_path = '../test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not test:
            for dataset in self.model.keys():
                self.model[dataset].eval()
                self.model[dataset].out_layer.eval()
            self.server_model.eval()

            
            train_energy ={}
            # (1) stastic on the train set
            for _client_id, data_loader in tqdm(train_loader.items(), desc="stastic on the train set", unit="client"):
                dataset = _client_id[9:]
           
                print(dataset)
                if (dataset in self.bce_list):
                    flag = True
                    self.anomaly_criterion = nn.BCELoss(reduction='none')
                else:
                    flag = False
                    self.anomaly_criterion = nn.MSELoss(reduce=False)
                client_id = dataset #_client_id[:8]
                energy = []
                attens_energy = []
                with torch.no_grad():
                    for i, (batch_x, batch_y,mask_bool) in enumerate(data_loader):
                        batch_x = batch_x.float().to(self.device[client_id])
                        if flag:
                            batch_y = batch_y.float().to(self.device[client_id])
                      
                          
                        self.model[dataset].to(batch_x.device)
                        client_A = self.model[dataset].forward(batch_x)
                        client_device = client_A.device
                        self.server_model = self.server_model.to(client_device)
                        server_B = self.server_model.Forward2(self.server_model, client_A)
                        outputs = self.model[dataset].Forward_3(self.model[dataset].out_layer,server_B,flag)
                        # criterion
                        if flag==False:
                            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                            # print('---',self.anomaly_criterion(batch_x, outputs).shape,self.anomaly_criterion(batch_x, outputs))
                        else:
                            score = torch.mean(self.anomaly_criterion(batch_y, outputs), dim=-1)
                            # print(self.anomaly_criterion(batch_y, outputs).shape,self.anomaly_criterion(batch_y, outputs))

                        score = score.detach().cpu().numpy()
                        energy.append(score)
                        del batch_x, batch_y,score

                attens_energy = np.concatenate(energy, axis=0).reshape(-1)
                tmp_train_energy = np.array(attens_energy)
                train_energy[client_id] = tmp_train_energy
            

            # (2) find the threshold
            attens_energy = []
            test_labels = {}
            test_energy = {}
            for _client_id, tes_data_loader in tqdm(test_loader.items(), desc="find the threshold", unit="client"):
                dataset = _client_id[9:]
                print(dataset)
                if (dataset in self.bce_list):
                    flag = True
                    self.anomaly_criterion = nn.BCELoss(reduction='none')
                else:
                    flag = False
                    self.anomaly_criterion = nn.MSELoss(reduce=False)
                client_id = dataset
                energy = []
                tmp_labels = []
                attens_energy = []
                result_dict={}
                
                for i, (batch_x, batch_y,mask_bool) in enumerate(tes_data_loader):
                    batch_x = batch_x.float().to(self.device[client_id])
                    if flag:
                        batch_y = batch_y.float().to(self.device[client_id])
                  
                    self.model[dataset].to(batch_x.device)
                    client_A = self.model[dataset].forward(batch_x)
                    client_device = client_A.device
                    self.server_model = self.server_model.to(client_device)
                    server_B = self.server_model.Forward2(self.server_model, client_A)
                    outputs = self.model[dataset].Forward_3(self.model[dataset].out_layer,server_B,flag)

                            
                    if flag==False:
                        score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                    else:
                        score = torch.mean(self.anomaly_criterion(batch_y.unsqueeze(-1), outputs), dim=-1)

                    score = score.detach().cpu().numpy()
                    energy.append(score)
                    tmp_labels.append(batch_y.cpu())

                    del batch_x, batch_y,score
                self.model[dataset].cpu()

                tmpt_labels = np.concatenate(tmp_labels, axis=0).reshape(-1)
                test_labels[client_id] = np.array(tmpt_labels)

                attens_energy = np.concatenate(energy, axis=0).reshape(-1)
                tmp_test_energy = np.array(attens_energy)
                test_energy[client_id] = tmp_test_energy
                
            combined_energy = {client_id: np.concatenate([train_energy[client_id], test_energy[client_id]], axis=0)
                    for client_id in train_energy.keys()}
        
            thresholds = {client_id: np.percentile(combined_energy[client_id], 100 - ast.literal_eval(self.args.anomaly_ratio)[client_id])
                for client_id in combined_energy.keys()}
            
     
            
            
            # (3) evaluation on the test set
            pred = {}
            gt = {}
            pate,pate_f1 = 0,0
            for _client_id, tes_data_loader in tqdm(test_loader.items(), desc="evaluation", unit="client"):
                client_id = _client_id[9:]
                dataset = client_id

                tmp_test_labels = test_labels[client_id]  
                gt[client_id] = tmp_test_labels.astype(int)
           
                pred[client_id] = (test_energy[client_id] > thresholds[client_id]).astype(int)        


            # (4) detection 
            total_result = {}
            total_result_new = {}
            for client_id, tes_data_loader in gt.items():
                tmp_gt, tmp_pred = adjustment(gt[client_id], pred[client_id])
                client_pred = np.array(tmp_pred)
                client_gt = np.array(tmp_gt)

                if len(set(client_gt)) > 1:
                    roc_auc = roc_auc_score(client_gt, client_pred)
                else:
                    # print(set(client_gt))
                    print("Only one class present in y_true. Skipping ROC AUC calculation.")
                
             
                roc_auc = roc_auc_score(client_gt, client_pred)*100 ####
                accuracy = accuracy_score(client_gt, client_pred)*100
                precision, recall, f_score, support = precision_recall_fscore_support(client_gt, client_pred, average='binary')
                precision, recall, f_score = precision*100, recall*100, f_score*100
                print(client_id, " --Accuracy: {:0.2f}, Precision: {:0.2f}, Recall: {:0.2f}, rou_auc: {:0.2f} F-score: {:0.2f}, ".format(
                    accuracy, precision,recall,
                    roc_auc, f_score))
                total_result[client_id] = {
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F-score': f_score,
                    'roc_auc':roc_auc,
                 
                }
            # Calculate and print mean values
            mean_accuracy = np.mean([result['Accuracy'] for result in total_result.values()])
            mean_precision = np.mean([result['Precision'] for result in total_result.values()])
            mean_recall = np.mean([result['Recall'] for result in total_result.values()])
            mean_f_score = np.mean([result['F-score'] for result in total_result.values()])
            mean_roc_auc = np.mean([result['roc_auc'] for result in total_result.values()])
          

            print("Mean Acc: {:0.2f}, Mean P: {:0.2f}, Mean R: {:0.2f}, Mean Roc: {:0.2f}, Mean F-score: {:0.2f}".format(
                mean_accuracy, mean_precision, mean_recall, mean_roc_auc, mean_f_score))

            return
        

