# Unity is Strength: Efficient Federated Anomaly Detection for Cross-domain Time Series with Sharded LLM [Scalable Data Science] [Under Review]
This repository provides the implementation of _Unity is Strength: Efficient Federated Anomaly Detection for Cross-domain Time Series with Sharded LLM [Scalable Data Science]_, called _FAST-MAD_ below.


## Installation
Install Python 3.8, PyTorch >= 1.3.1.
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


## Train and evaluate.
We provide the experiment script under the folder ./scripts. You can reproduce the experiment results as follows:
```
bash ./scripts/main_tes.sh
```
