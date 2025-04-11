# FAST-MAD: Efficient Federated Anomaly Detection for Cross-domain Time Series with Sharded LLMs [Under Review]


## Environment
Run the following script for environment configuration.
```
pip install -r requirements.txt
```


## Datasets
### SMD, PSM, SMAP, and MSL:
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

We implement our model with the PyTorch framework on NVIDIA RTX 3090 GPU. The LLM weight is downloaded from Huggingface. We first split the time series into consecutive non-overlapping segments by sliding window[1]. The batch size 64. Adam is adopted for optimization. We employ GPT2 as the LLM, where the first three layers are separated into the modality translator module, and the remaining seven stacked transformer blocks are located in the server. The number of clients is set to `6` by default, each with a local dataset (e.g., UCR, SWaT, SMD, MSL, PSM, SMAP) from different domains. For the UCR dataset, we use its subset DISTORTED1sddb40. When comparing the computational overhead (FLOPS) of the model and the baselines, the experimental results are consistently calculated with a batch size of 32. We follow the same data processing and train-validation-test set split protocol and the threshold settings used in TimesNet. For all baseline methods and clients, a non-overlapping sliding window of length 100 is applied to process the time series data, followed by standardization using StandardScaler. The threshold 𝑟 for UCR, SMAP, SMD, MSL, PSM, and SWaT is set to `0.5`, `2`, `0.5`, `2`, `1`, and `1`, respectively. The learning rate is set individually for each dataset as follows: `0.0001` for SMD, PSM, SWAT, SMAP, and UCR, and `0.00001` for MSL. The hidden state dimension of the model is set to `1280`.


[1] Datar M, Gionis A, Indyk P, et al. Maintaining stream statistics over sliding windows[J]. SIAM journal on computing, 2002, 31(6): 1794-1813.

## Baseline settings
**DeepSVDD**：We use the Adam optimizer with default settings and apply Batch Normalization. The two-phase learning rate schedule is adopted: `1e-4` for searching and `1e-5` for fine-tuning. 

**TranAD**：We use the AdamW optimizer with an initial learning rate of `0.01` (meta learning rate `0.02`) following the paper. Key hyperparameters include: feed-forward layers per encoder `2`, hidden units per layer `64`, and dropout rate `0.1`.

**MICN**: We train MICN using L2 loss and the Adam optimizer with an initial learning rate of `1e-3`. Batch size is set to `32`, and early stopping is applied after 3 epochs without validation loss improvement.

**DLinear**: We use a non-overlapping sliding window with a length of `100`, and the patch length is set to `10`. The Adam optimizer is used. The learning rate is set to `1e-5`.

**Transformer**: It is trained using the Adam optimizer with L2 loss, an initial learning rate of `1e-4`, and a batch size of `32`. 

**Autoformer**: It is trained with L2 loss using the Adam optimizer with an initial learning rate of `1e-4`. Batch size is set to `32`. The training process is early stopped within 10 epochs.

**Crossformer**: We roll the whole set with stride `1` to generate different input-output pairs. The learning rate is set to `1e-5`, and the Adam optimizer is used

**iTransformer**: We use the Adam optimizer with an initial learning rate in `The learning rate is set to `1e-5`. The batch size is set to `32`. The dimension of series representations `D` is set to `128`.

**Fedformer**: The model is trained using the Adam optimizer with a learning rate of `1e-4`. The batch size is set to `32`. An early stopping counter is employed to stop training after three epochs if no loss degradation on the validation set is observed.

**UniTS**: We use a batch size of `32`. The learning rate starts at `3.2e-2` and is adjusted with a multi-step decayed schedule following the papper.

**TimesNet**: We split the dataset into consecutive non-overlapping segments using a sliding window. The batch size is set to `32`, and the Adam optimizer is used. The learning rate is set to `1e-5`.

**Anomaly Transformer**: The channel number of hidden states `d_model` is set to `512`, and the number of heads `h` is set to `8`. We use the Adam optimizer with an initial learning rate of `1e-4`. The training process is early stopped within `10` epochs with a batch size of `32`.

**FPT**: We use a non-overlapping sliding window with a length of `100`, and the patch length is set to `10`. The Adam optimizer is used. The learning rate is set to `1e-4`.

**PeFAD**: The patch length and batch size are set to `10` and `32`, respectively. Adam is adopted for optimization. The Adam optimizer is used. The learning rate is set individually for each dataset as follows: `0.0001` for SMD, PSM, and SWaT; `0.00001` for MSL; and `1e-5` for UCR and SMAP.





