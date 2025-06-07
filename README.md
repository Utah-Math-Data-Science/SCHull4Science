# SCHull in Scientific Application
This repository contains the code used for the experiments presented in the paper [A Theoretically-Principled Sparse, Connected, and Rigid Graph Representation of Molecules](https://openreview.net/pdf?id=OIvg3MqWX2), selected for an [Oral presentation at ICLR 2025](https://iclr.cc/virtual/2025/oral/31862).
<p align="center">
<img src="https://github.com/Utah-Math-Data-Science/SCHull4Science/blob/main/SCHull_fig.png" alt="Description of image" width="70%">
</p>

## Environment
All experiments were conducted using Python 3.9.20 within a [Conda 24.11.3](https://anaconda.org/anaconda/conda/files?page=2&sort=ndownloads&sort_order=asc&type=conda&version=24.11.3) environment, with CUDA 12.4 support on NVIDIA RTX 3090 GPUs. All required packages are specified in the [`requirements.txt`](https://github.com/Utah-Math-Data-Science/SCHull4Science/blob/main/requirements.txt) file.
 
## Dataset
### Dataset Files
All datasets are saved as `.mdb` files. Please download the dataset from the provided [link](https://drive.google.com/file/d/1oCFu8EN1ZbtMMiFqCP-p4OaqQkRn4Elx/view?usp=sharing) or using
```
pip install gdown
gdown https://drive.google.com/uc?id=1oCFu8EN1ZbtMMiFqCP-p4OaqQkRn4Elx
```
and unzip it before running the experiments. The data folders are organized as follows:
```bash
├── Data
│     │           
│     ├── Reaction-EC
│     │       │
│     │       ├── train
│     │       │     ├── data.mdb
│     │       │     ├── lock.mdb
│     │       │     ...
│     │       ├── val 
│     │       ...
│     ├── FoldData
│     ...               
```
### Data Pre-processing
When the code is run for the first time, it will automatically pre-process the data, which includes constructing the original node features and the [SCHull](https://openreview.net/pdf?id=OIvg3MqWX2) graph using 
```
import SCHull; schull = SCHull.SCHull()
schull.get_schull
```
Each pre-processing step takes less than 10 minutes to complete, with approximately 40% of the time spent on constructing the SCHull graph.


## Experiments
The experimental codes are organized as:
```bash
├── SCHull4Science
│       │           
│       ├── SCHull
│       │      ├── SCHull.py
│       │      ...
│       ├── models
│       │      ├── pronet.py
│       │      ...
│       ├── dataset
│       │      ├── fold_dataset.py
│       │      ...
│       ...
│       ├── main_react.py
│       │
│       ├── main_react.py
│     ...
```
To run the experiments, one approach is to `cd` to the `SCHull4Science` directory and execute the following command:

### Reaction Classification
```bash

python main_react.py --data_path <PATH_to_Data/Reaction-EC> --exp_name <Experiment_Name>

```

### Fold Classification
```bash

python main_fold.py --data_path <PATH_to_Data/FoldData> --exp_name <Experiment_Name>

```

### LBA Prediction
```bash

python main_lba.py --data_path <PATH_to_Data/LBA-split-by-sequence-identity-30> --exp_name <Experiment_Name>

```

## Claim
This repository incorporates components of code from the [DIG](https://github.com/divelab/DIG).
