# Unity is Strength: Efficient Federated Anomaly Detection for Cross-domain Time Series with Sharded LLM [Scalable Data Science] [Under Review]


## Environment
Run the following script for environment configuration.
```
pip install -r requirements.txt
```


## Datasets
### SMD, PSM, and MSL:
SMD, PSM, and MSL can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1gisthCoE-RrKJ0j3KPV7xiibhHWT9qRm).  
### SWaT dataset:
Please refer to https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/ for SWaT.  
### UCR dataset:
UCR can be downloaded from its official Archive, link: https://www.cs.ucr.edu/~eamonn/time_series_data_2018.


## Train and evaluate
We provide the experiment scripts in the folder ./scripts. The experiment results can be reproduced as follows.
```
bash ./scripts/main_tes.sh
```
## Implementation Details
Please refer to [implementation details](ImplemetationDetail.md)
