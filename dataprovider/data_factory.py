from data_provider.data_loader import PSMSegLoader, MSLSegLoader, SMDSegLoader, SWATSegLoader,UCRSegLoader#,SMAPSegLoader
from torch.utils.data import DataLoader
import numpy as np
import os
import pandas as pd
from data_provider.ucr import UCR
from merlion.utils import TimeSeries
from merlion.transform.normalize import MeanVarNormalize 
import ast

def other_datasets(time_series, meta_data):
    train_time_series_ts = TimeSeries.from_pd(time_series[meta_data.trainval])
    test_time_series_ts = TimeSeries.from_pd(time_series[~meta_data.trainval])
    train_labels = TimeSeries.from_pd(meta_data.anomaly[meta_data.trainval])
    test_labels = TimeSeries.from_pd(meta_data.anomaly[~meta_data.trainval])
    mvn = MeanVarNormalize()
    mvn.train(train_time_series_ts + test_time_series_ts)
    # salesforce-merlion==1.1.1
    bias, scale = mvn.bias, mvn.scale

    train_time_series = train_time_series_ts.to_pd().to_numpy()

    # train_data = (train_time_series - bias) / scale
    test_time_series = test_time_series_ts.to_pd().to_numpy()
    # test_data = (test_time_series - bias) / scale

    train_data = train_time_series
    test_data = test_time_series
    train_labels = train_labels.to_pd().to_numpy()
    test_labels = test_labels.to_pd().to_numpy()

    return train_data, test_data, train_labels, test_labels


def Distribute_data(dataset_name, root_path, flag,n,r=1):
    if dataset_name == 'SMD':
        train_data = np.load(os.path.join(root_path, "SMD/SMD_train.npy"))
        train_labels = train_data
        test_data = np.load(os.path.join(root_path, "SMD/SMD_test.npy"))
        test_labels = np.load(os.path.join(root_path, "SMD/SMD_test_label.npy"))
       
    elif dataset_name == 'MSL':
        train_data = np.load(os.path.join(root_path, "MSL/MSL_train.npy"))
        train_labels = train_data
        test_data = np.load(os.path.join(root_path, "MSL/MSL_test.npy"))
        test_labels = np.load(os.path.join(root_path, "MSL/MSL_test_label.npy"))
       
        
    elif dataset_name == 'PSM':
        data = pd.read_csv(os.path.join(root_path, 'PSM/train.csv'))
        data = data.values[:, 1:]
        train_data = np.nan_to_num(data)
        train_labels = train_data
        test_data =  pd.read_csv(os.path.join(root_path, 'PSM/test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        test_labels = pd.read_csv(os.path.join(root_path, 'PSM/test_label.csv')).values[:, 1:]
       
        
    elif dataset_name == 'SWAT':
        data = pd.read_csv(os.path.join(root_path, 'SWaT/swat_train2.csv'))
        train_data = data.values[:, :-1]
        train_labels = train_data
        test_data = pd.read_csv(os.path.join(root_path, 'SWaT/swat2.csv'))
        test_labels = test_data.values[:, -1:]
        test_data = test_data.values[:, :-1]
        # r= 0.0003
       

    elif dataset_name == 'UCR':
        dt = UCR()
        train_data_list = []
        test_data_list = []
        train_labels_list = []
        test_labels_list = []

        for i in range(n):
            time_series, meta_data = dt[i]
            
            train_data, test_data, train_labels, test_labels = other_datasets(time_series, meta_data)
            train_data_list.append(train_data)
            test_data_list.append(test_data)
            train_labels_list.append(train_labels)
            test_labels_list.append(test_labels)
            train_data = np.concatenate(train_data_list, axis=0)
            train_labels = np.concatenate(train_labels_list, axis=0)
            test_data = np.concatenate(test_data_list, axis=0)
            test_labels = np.concatenate(test_labels_list, axis=0)
        if r<1:
            train_data = train_data[:int(r * len(train_data))]
            train_labels = train_labels[:int(r * len(train_labels))]
           
        
    return train_data, train_labels, test_data, test_labels


def data_provider(args, flag, shared):
    data_dict = {
    'SMD': SMDSegLoader,
    'MSL': MSLSegLoader,
    'PSM': PSMSegLoader,
    'SWAT': SWATSegLoader,
    'UCR': UCRSegLoader
    }

    if flag == 'test':
        shuffle_flag = False
    else:
        shuffle_flag = True
   
    drop_last = False              

    dataset_names = ast.literal_eval(args.dataset_names)
    steps = ast.literal_eval(args.steps)

    client_data = {}
    client_test_data = {}
    client_test_labels = {}
    for i, dataset_name in enumerate(dataset_names):
        
        train_data, train_labels, test_data, test_labels = Distribute_data(dataset_name=dataset_name, root_path=args.root_path, flag=flag,n=args.n_ucr)
        client_data[f'client_{i+1}_{dataset_name}'] = {'X': train_data, 'Y': train_labels}
        client_test_data[f'client_{i+1}_{dataset_name}'] = {'test_data': test_data,'test_labels': test_labels}

    client_dataset_dict = {}
    # print("step: ",steps)
    for client_id, client_train_data in client_data.items():

        client_X = client_data[client_id]['X']
        client_Y = client_data[client_id]['Y']
        client_test_X = client_test_data[client_id]['test_data']
        client_test_Y = client_test_data[client_id]['test_labels']

        client_test_Y = client_test_data[client_id]['test_labels']
        # print(client_id,client_X.shape,client_Y.shape,client_test_X.shape, client_test_Y.shape) 
        dataset_name = client_id[9:]
        step = steps[dataset_name]
        Data = data_dict[dataset_name]
        client_dataset_dict[client_id] = Data(
            data=client_X,
            train_labels=client_Y,
            test_data=client_test_X,
            test_labels=client_test_Y,
            root_path=args.root_path,  
            win_size=args.seq_len,
            flag=flag,
            patch_len=args.patch_len,
            patch_stride=args.patch_stride,
            mask_ratio=args.mask_ratio,
            step=step,
            seleck_k=args.seleck_k,
            periods_num = args.periods_num,
            multi_resolution = args.multi_resolution
        )
    
    client_loader_dict = {}
    for client_id, client_train_data in client_data.items():
        client_loader_dict[client_id] = DataLoader(
            client_dataset_dict[client_id],
            batch_size=args.local_bs,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )  
    
    return client_dataset_dict, client_loader_dict
