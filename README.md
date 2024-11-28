# Unity is Strength: Efficient Federated Anomaly Detection for Cross-domain Time Series with Sharded LLM [Scalable Data Science] [Under Review]
This repository provides the implementation of _Unity is Strength: Efficient Federated Anomaly Detection for Cross-domain Time Series with Sharded LLM [Scalable Data Science]_, called _FAST-MAD_ below.


## Installation
This code is based on Python 3.8, PyTorch 1.3.1, the requirements are written in requirements.txt.
```
pip install -r requirements.txt
```


## Dataset
### SMD, PSM, and MSL dataset:
You can download SMD, PSM, and MSL from [Google Drive](https://drive.google.com/drive/folders/1gisthCoE-RrKJ0j3KPV7xiibhHWT9qRm).  
### SWaT dataset:
For the SWaT dataset, you can apply for it by following its official tutorial, link: https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/.  
### UCR dataset:
For the UCR dataset, you can apply for it by following its official Archive, link: https://www.cs.ucr.edu/~eamonn/time_series_data_2018.


## Train and evaluate
We provide the experiment script under the folder ./scripts. You can reproduce the experiment results as follows:
```
bash ./scripts/main_tes.sh
```
## Implementation Details
We implement our model with the PyTorch framework on NVIDIA RTX 3090 GPU. The LLM weight is downloaded from Huggingface. We first split the time series into consecutive non-overlapping segments by sliding window~\cite{datar2002maintaining}. The batch size 64. Adam is adopted for optimization. We employ GPT2 as the LLM, where the first three layers are separated into the modality translator module, and the remaining seven stacked transformer blocks are located in the server. The number of clients is set to 5 by default, each with a local dataset (e.g., UCR, SWaT, SMD, MSL, PSM) from different domains. For the UCR dataset, we use its subset DISTORTED1sddb40. When comparing the computational overhead (FLOPS) of the model and the baselines, the experimental results are consistently calculated with a batch size of 32.
