## Implementation Details

We implement our model with the PyTorch framework on NVIDIA RTX 3090 GPU. The LLM weight is downloaded from Huggingface. We first split the time series into consecutive non-overlapping segments by sliding window[1]. The batch size is set to 64. Adam is adopted for optimization. We employ GPT2 as the LLM, where the first three layers are separated into the modality translator module, and the remaining seven stacked transformer blocks are located in the server. The number of clients is set to 5 by default, each with a local dataset (e.g., UCR, SWaT, SMD, MSL, PSM) from different domains. For the UCR dataset, we use its subset DISTORTED1sddb40. When comparing the computational overhead (FLOPS) of the model and the baselines, the experimental results are consistently calculated with a batch size of 32.

[1] Datar M, Gionis A, Indyk P, et al. Maintaining stream statistics over sliding windows[J]. SIAM journal on computing, 2002, 31(6): 1794-1813. 
