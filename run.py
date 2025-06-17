import argparse
import os
import torch

from exp.exp_anomaly_detection import Exp_Anomaly_Detection

import random
import numpy as np
from data_provider.data_factory import data_provider

import torch.multiprocessing as mp
if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='FASTMAD')

    parser.add_argument('--local_bs', type=int, required=True, default=256, help='local batch size')
    parser.add_argument('--local_epoch', type=int, required=True, default=1, help='local training epoch')

    parser.add_argument('--patch_len', type=int, required=False, default=10, help='the length of patch')
    parser.add_argument('--patch_stride', type=int, required=False, default=10, help='the stride of patch')
    parser.add_argument('--multi_resolution', type=int, required=False, default=1, help='the length of patch')

    parser.add_argument('--mask_ratio', type=float, required=False, default=0.2, help='mask ratio of patch')
    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='anomaly_detection')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='False', help='model id')
    parser.add_argument('--model', type=str, required=True, default='GPT4TS')

    # data loader
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoint/', help='location of model checkpoints')
    parser.add_argument('--seq_len', type=int, default=100, help='input sequence length')
    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=1280, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=16, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--save_path', type=str, required=False, default='save_path', help='save_path')
    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=2, help='train epochs')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=2, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='3,2,1,0', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # patching
    # parser.add_argument('--patch_size', type=int, default=1)
    # parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--gpt_layers_client', type=int, default=3)
    parser.add_argument('--gpt_layers', type=int, default=10)

    parser.add_argument('--ln', type=int, default=0)
    parser.add_argument('--mlp', type=int, default=0)
    parser.add_argument('--weight', type=float, default=0)
    parser.add_argument('--percent', type=int, default=5)

    parser.add_argument('--param_loss_coef', type=float, default=0.1)
    parser.add_argument('--mp_worker_num', type=int, default=8)
    parser.add_argument('--n_ucr', type=int, default=2)
    parser.add_argument('--seleck_k', type=int, default=3)
    parser.add_argument('--periods_num', type=int, default=6)


    parser.add_argument('--dataset_to_layers', type=str, default='{"SMD": 3, "MSL": 1, "PSM": 3, "SWAT": 1, "UCR": 3}')
    parser.add_argument('--learning_rate', type=str, default='{"SMD": 0.0001, "MSL": 0.00001, "PSM": 0.0001, "SWAT": 0.0001, "UCR": 0.0001}')
    parser.add_argument('--dataset_ids', type=str, default="{'K1':'SMD', 'K2':'MSL', 'K3':'PSM', 'K4':'SWAT', 'K6':'UCR'}")
    parser.add_argument('--dataset_c_out_dict', type=str, default='{"SMD": 38, "MSL": 55, "PSM": 25, "SWAT": 51, "UCR": 1}')
    parser.add_argument('--anomaly_ratio', type=str, default='{"SMD": 0.5, "MSL": 2, "PSM": 1, "SWAT": 1,"UCR": 0.5}')
    parser.add_argument('--dataset_names', type=str, default='["UCR","SMD","MSL", "PSM", "SWAT"]')
    parser.add_argument('--steps', type=str, default='{"SMD": 100, "MSL": 10, "PSM": 10, "SWAT": 10,  "UCR": 10}')
    parser.add_argument('--dataset_names_otherone', type=str, default='[]')
    parser.add_argument('--mask_matrix_arg', type=str, default='["MSL"]')
    parser.add_argument('--dataset_device', type=str, default='{"UCR": 3, "SMD": 2, "MSL": 3, "PSM": 1, "SWAT": 2')
    parser.add_argument('--main_layer', type=str, default='{"SMD": 10, "MSL": 10, "PSM": 10, "SWAT": 10, "UCR": 0}')


    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        print('devices:',device_ids)
        args.device_ids = [int(id_) for id_ in device_ids]
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
        # args.gpu = args.device_ids[3]
        # print(f"Using  GPU with device ID: {args.gpu}")


    Exp = Exp_Anomaly_Detection
  

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}'.format(args.model_id,  args.model)
            exp = Exp(args)  # set experiments
                
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting=setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,  
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
