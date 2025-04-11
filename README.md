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

## Implementation Details

We implement our model with the PyTorch framework on NVIDIA RTX 3090 GPU. The LLM weight is downloaded from Huggingface. We first split the time series into consecutive non-overlapping segments by sliding window[1]. The batch size 64. Adam is adopted for optimization. We employ GPT2 as the LLM, where the first three layers are separated into the modality translator module, and the remaining seven stacked transformer blocks are located in the server. The number of clients is set to 5 by default, each with a local dataset (e.g., UCR, SWaT, SMD, MSL, PSM) from different domains. For the UCR dataset, we use its subset DISTORTED1sddb40. When comparing the computational overhead (FLOPS) of the model and the baselines, the experimental results are consistently calculated with a batch size of 32.

[1] Datar M, Gionis A, Indyk P, et al. Maintaining stream statistics over sliding windows[J]. SIAM journal on computing, 2002, 31(6): 1794-1813.

## Baseline details
**DeepSVDD**：We use the Adam optimizer with default settings and apply Batch Normalization. Network weights are initialized using Glorot uniform, while DeepSVDD uses pre-trained DCAE encoder weights. A two-phase learning rate schedule is adopted: `1e-4` for searching and `1e-5` for fine-tuning. 
**TranAD**：We use the AdamW optimizer with an initial learning rate of `0.01` (meta learning rate `0.02`) and a step scheduler with step size `0.5`. Key hyperparameters include: window size `10`, transformer encoder layers `1`, feed-forward layers per encoder `2`, hidden units per layer `64`, and dropout rate `0.1`.
**MICN**: We train MICN using L2 loss and the Adam optimizer with an initial learning rate of `1e-3`. Batch size is set to `32`, and early stopping is applied after 3 epochs without validation loss improvement. Hyperparameter `i` is set to `{12, 16}`, and input length is fixed to `96` for all datasets (except `36` for ILI).


