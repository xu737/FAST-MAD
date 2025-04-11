# FAST-MAD: Efficient Federated Anomaly Detection for Cross-domain Time Series with Sharded LLMs [Under Review]


## Environment
Run the following script for environment configuration.
```
pip install -r requirements.txt
```


## Datasets
### SMD, PSM, and MSL:
SMD, PSM, and MSL can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1gisthCoE-RrKJ0j3KPV7xiibhHWT9qRm).  
### SWaT dataset:
Please refer to https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/ for SWaT download.  
### UCR dataset:
UCR can be downloaded from its official [Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018).


## Train and evaluate
We provide the experiment scripts in the folder ./scripts. The experiment results can be reproduced as follows.
```
bash ./scripts/main_tes.sh
```

## Baseline details
### DeepSVDD
We use the Adam optimizer with default settings and apply Batch Normalization. Network weights are initialized using Glorot uniform, while DeepSVDD uses pre-trained DCAE encoder weights. A two-phase learning rate schedule is adopted: `1e-4` for searching and `1e-5` for fine-tuning. DCAE is trained for 250+100 epochs, and DeepSVDD for 150+100 epochs. Leaky ReLU is used with a leakiness of `0.1`.
### TranAD
We use the AdamW optimizer with an initial learning rate of `0.01` (meta learning rate `0.02`) and a step scheduler with step size `0.5`. Key hyperparameters include: window size `10`, transformer encoder layers `1`, feed-forward layers per encoder `2`, hidden units per layer `64`, and dropout rate `0.1`.


