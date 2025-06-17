import os
import torch

from models import GPT4TS
import ast

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'GPT4TS': GPT4TS,
        }
        self.device = self._acquire_device()
        print('--',self.device)
        self.model = self._build_model()
 

    def _build_model(self):
        raise NotImplementedError
        return None

    
    def _acquire_device(self):
        devices = {}
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = '3,2,1,0'
            list = [0,1,2,3,1,2]
            # dataset = {"UCR": 1, "SMD": 1, "MSL": 1, "PSM": 1, "SWAT": 1}
            dataset = ast.literal_eval(self.args.dataset_device)
            for k,v in dataset.items():  # Assuming you want to use 4 GPUs
                # devices[f'client_{i+1}'] = torch.device(f'cuda:{list[i]}')
                devices[k] = torch.device(f'cuda:{v}')
                print(f'Use GPU: {k} -> cuda:',{devices[k]})
        else:
            devices['cpu'] = torch.device('cpu')
            print('Use CPU')
        return devices

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
