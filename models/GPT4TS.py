#8766
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import ast
from transformers import GPT2ForSequenceClassification
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import BertTokenizer, BertModel
from einops import rearrange
# from layers.Embed import DataEmbedding, DataEmbedding_wo_time
# from data_provider.patch_mask import *
# from models.AnomalyTransformer import AnomalyTransformer

class Model(nn.Module):
    
    def __init__(self, dataset_id, configs):
        super(Model, self).__init__()
        self.is_ln = configs.ln
        self.task_name = configs.task_name
        # self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        # self.patch_size = configs.patch_len  #1
        # self.stride = configs.patch_stride
        self.seq_len = configs.seq_len
        self.patch_len = configs.patch_len
        self.d_ff = configs.d_ff
        self.gpu = configs.gpu
        # self.gptLarge = configs.gptLarge
        # self.full_tuning = configs.full_tuning
        # self.adapter = configs.adapter
        # self.efficient_tuning = configs.efficient_tuning
        # self.effi_layer = configs.effi_layer
        self.model_id = configs.model_id
        # self.tune_layers = configs.tune_layers
        self.model = configs.model
        dataset_c_out_dict = ast.literal_eval(configs.dataset_c_out_dict)
        self.dataset_to_layers = ast.literal_eval(configs.dataset_to_layers)
        self.sigmoid = nn.Sigmoid() 
        self.dataset_id=dataset_id
        self.mask_matrix_arg = ast.literal_eval(configs.mask_matrix_arg)
        self.device={}
        dataset_device = ast.literal_eval(configs.dataset_device)
        for k,v in dataset_device.items():  # Assuming you want to use 4 GPUs
            self.device[k] = torch.device(f'cuda:{v}')
        self.main_layer = ast.literal_eval(configs.main_layer)
        print('main_layer',self.main_layer)
        self.c_out = dataset_c_out_dict[dataset_id]

     
        self.gpt2 = GPT2Model.from_pretrained('/home/data/xrh/FL/AD_FL/gpt2large', output_attentions=True, output_hidden_states=True)

        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]  
        
        self.gpt2_client = self.gpt2.h[:configs.gpt_layers_client] #nn.ModuleList(self.gpt2.h[:configs.gpt_layers_client])
        self.gpt2_main = self.gpt2.h[configs.gpt_layers_client:configs.gpt_layers] #nn.ModuleList(self.gpt2.h[configs.gpt_layers_client:configs.gpt_layers])

        tuning_layers = self.dataset_to_layers.get(dataset_id, 1)
        self.set_parameter_requires_grad(self.gpt2_client,tuning_layers)
     

        self.set_parameter_requires_grad(self.gpt2_main,self.main_layer[dataset_id],main=True)

    
        self.mask_matrix = nn.ParameterDict()
        for i,layer in enumerate(self.gpt2_client):
            self.mask_matrix[str(i)] =  nn.Parameter(torch.randn([64, 100, 1280]), requires_grad=True)
            # self.mask_matrix[str(i)] =  nn.Parameter(torch.randn([3840]), requires_grad=True)


        if self.task_name == 'anomaly_detection':
            self.ln_proj = nn.LayerNorm(configs.d_ff)
            self.out_layer = nn.Linear(
                configs.d_ff, 
                self.c_out, 
                bias=True)

    def forward(self, x_enc):
        # x_enc.requires_grad_(True)
        bs, num_patch_patch_length, nvars = x_enc.shape
        seg_num =  self.patch_len
        x_enc = rearrange(x_enc, 'b (n s) m -> b n s m', s=seg_num)
        means = x_enc.mean(2, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=2, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        self.stdev_1 = stdev
        self.means_1 = means
        x_enc = rearrange(x_enc, 'b n s m -> b (n s) m')
        
        self.enc_out = torch.nn.functional.pad(x_enc, (0, 1280-x_enc.shape[-1]))       
        # client_out = self.gpt2_client(inputs_embeds=self.enc_out).last_hidden_state
        self.client_out = self.enc_out
        if self.client_out.shape[0]==64:
            for i,layer in enumerate(self.gpt2_client): # layer.requires_grad==True and 
                # layer.attn.c_attn.weight=layer.attn.c_attn.weight*self.mask_matrix[str(i)] ###todo
                
                if self.dataset_id in self.mask_matrix_arg:
                    self.client_out = layer(self.client_out)[0]*self.mask_matrix[str(i)]
                else:
                    self.client_out = layer(self.client_out)[0]#*self.mask_matrix[str(i)] ####

        else:
            for i,layer in enumerate(self.gpt2_client): # layer.requires_grad==True and 
                self.client_out = layer(self.client_out)[0]
        client_x_A = self.client_out.clone().detach()
        client_x_A.requires_grad_(True)
        # self.client_out.requires_grad_(True)
        return client_x_A
    
    def Forward2(self, server_model, client_A):
        self.server_B = client_A
        # for layer in self:
        for layer in server_model.gpt2_main:
            self.server_B = layer(self.server_B)[0]

        server_B = self.server_B.clone().detach().requires_grad_(True)
        # server_B.retain_grad() 
        return server_B

    def Forward_3(self, client_model_B,server_outputs,flag=False):    
        outputs = server_outputs[:, :, :self.d_ff]
        dec_out = self.out_layer(outputs)
        
        # De-Normalization from Non-stationary Transformer
        seg_num =  self.patch_len
        dec_out = rearrange(dec_out, 'b (n s) m -> b n s m', s=seg_num)
        dec_out = dec_out * \
                    (self.stdev_1[:, :, 0, :].unsqueeze(2).repeat(
                        1, 1, seg_num, 1))
        dec_out = dec_out + \
                    (self.means_1[:, :, 0, :].unsqueeze(2).repeat(
                        1, 1, seg_num, 1))
        dec_out = rearrange(dec_out, 'b n s m -> b (n s) m')
        
        if flag:
            probs = self.sigmoid(dec_out)
            dec_out = probs#.squeeze(-1)
        return dec_out

    def Backward2(self, server_B, dB, client_A, optimizer_server):
        self.server_B.backward(dB) #,retain_graph=True
       
        server_dA = client_A.grad.clone().detach() ######
        optimizer_server.step()
        
        del self.server_B
        torch.cuda.empty_cache()

        return server_dA

    def Backward3(self, client_A, dA, optimizer_A,optimizer_mask,optimizer_for_weight):
        self.client_out.backward(dA)
        optimizer_A.step()
        optimizer_mask.step()
        optimizer_for_weight.step()

        del self.client_out
        torch.cuda.empty_cache()
        self.selected_keys = self.select_top_50_percent(self.mask_matrix)
        return self.selected_keys

    def set_parameter_requires_grad(self,model,tuning_layers,main=False):
        if main:
            self.effi_layer = tuning_layers
            if self.effi_layer == 0:
                print('---main_full---')
                for i, (name, param) in enumerate(model.named_parameters()):
                    param.requires_grad = True
                return
            elif self.effi_layer == 10:
                print('---main_layernorm---')
                for i, (name, param) in enumerate(model.named_parameters()):
                    if 'ln' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                return
            else:
                if self.effi_layer == 3:
                    print('---main_3---')
                    is_top_layer_param = lambda name: any(layer_name in name for layer_name in ['6', '5', '4', 'ln_f'])
                elif self.effi_layer == 2:
                    print('---main_2---')
                    is_top_layer_param = lambda name: any(layer_name in name for layer_name in ['6','5', 'ln_f'])
                elif self.effi_layer == 1:
                    print('---main_1---')
                    is_top_layer_param = lambda name: any(layer_name in name for layer_name in ['6'])
                is_special_param = lambda name: any(layer_name in name for layer_name in ['wpe'])

                for i, (name, param) in enumerate(model.named_parameters()):
                    if is_top_layer_param(name) or is_special_param(name):
                        # if 'ln' in name:
                        #     param.requires_grad = False
                        # else:
                        #     param.requires_grad = True
                        if 'att' in name:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    else:
                        param.requires_grad = False
        else:
            # print('efficient-tuning')
            self.effi_layer = tuning_layers
            if self.effi_layer == 0:
                print('---0---')
                is_top_layer_param = lambda name: any(layer_name in name for layer_name in ['ln'])

            elif self.effi_layer == 3:
                print('---3---')
                is_top_layer_param = lambda name: any(layer_name in name for layer_name in ['0', '1', '2', 'ln_f'])
            elif self.effi_layer == 2:
                print('---2---')
                is_top_layer_param = lambda name: any(layer_name in name for layer_name in ['2','1', 'ln_f'])
            elif self.effi_layer == 1:
                print('---1---')
                is_top_layer_param = lambda name: any(layer_name in name for layer_name in ['2'])
            elif self.effi_layer == 10:
                print('---client_full_tuning---')
                for i, (name, param) in enumerate(model.named_parameters()):
                    param.requires_grad = True
                    return 
            is_special_param = lambda name: any(layer_name in name for layer_name in ['wpe'])

  
            for i, (name, param) in enumerate(model.named_parameters()):
                # print('name',name,is_top_layer_param(name))
                if is_top_layer_param(name) or is_special_param(name):
                    if 'ln' in name:
                        param.requires_grad = True
                    elif 'att' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    param.requires_grad = False

    def decompose_last_layer_svd(self,  k=5):
      
        weight  = self.gpt2_client[-1].attn.c_attn.weight#[:500,:500] #torch.Size([1280])
        # print('self.gpt2_client[-1].named_parameters()', self.gpt2_client[-1].named_parameters())
        W2,error = 0,0
        try:
            U, S, V = torch.svd(weight+torch.eye(weight.size(0), weight.size(1), device=weight.device))
            # weight += torch.eye(weight.size(0)) * 1e-6  
            # U, S, V = torch.linalg.svd(weight) #+torch.eye(weight.size(0), weight.size(1), device=weight.device), full_matrices=True
            U_k = U[:, :k]
            S_k = torch.diag(S[:k])
            V_k = V[:, :k]
            W1 = U_k @ S_k @ V_k.T  
      
            return W1, W2, error

        except torch._C._LinAlgError as e:
            # print(f"SVD failed: {e}")
            # print(f"SVD failed: {e}")
            return torch.zeros_like(weight), W2, error 

    def select_top_50_percent(self,mask_matrix):
        mask_counts = [] 
        for name, param in self.mask_matrix.items():
            mask = self.create_mask(param.data) 
            total_elements = mask.numel()
            num_ones = mask.sum().item()/total_elements  
            mask_counts.append((name, num_ones))
        # print('mask_counts',mask_counts)
        
        mask_counts.sort(key=lambda x: x[1], reverse=True)  
        top_50_percent = mask_counts[:len(mask_counts) // 2]  

        selected_keys = [name for name, _ in top_50_percent]
        return selected_keys
    def create_mask(self, param):
        # global_min, global_max = param.min(), param.max()
        # normalized_param = (param - global_min) / (global_max - global_min)
        # sigmoid_result = torch.sigmoid(normalized_param)
        sigmoid_result = torch.sigmoid(param)
        mask = (sigmoid_result > 0.5).float()
        return mask