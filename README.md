# SCHull in Scientific Application
This repository contains the code used for the experiments presented in the [SCHull](https://openreview.net/pdf?id=OIvg3MqWX2) paper, selected for an Oral presentation at ICLR 2025.

## Environment
All experiments were conducted using Python 3.9.20 within a Conda 24.11.3 environment, with CUDA 12.4 support on NVIDIA RTX 3090 GPUs. All required packages are specified in the `requirements.txt` file.
 
## Dataset
### Dataset Files
All datasets are saved as `.mdb` files. Please download the dataset from the provided [link](https://drive.google.com/drive/folders/15js5KZqXsEOZSdCmx52JpqwjQn7SOpo9?usp=drive_link) and unzip it before running the experiments. The data folders are organized as follows:
```bash
├── Data
│     │           
│     ├── Reaction-EC
│     │          ├── train
│     │          │      ├── data.mdb
│     │          │      ├── lock.mdb
│     │          ├── val
│     │          │     ...
│     │           ...
│     ├── FoldData
│     │          ...
│     │                    
│   ...
```
### Data Pre-processing
When the code is run for the first time, it will automatically pre-process the data, which includes constructing the original node features and the [SCHull](https://openreview.net/pdf?id=OIvg3MqWX2) graph using `schull.get_schull`. Each pre-processing step takes less than 10 minutes to complete, with approximately 40% of the time spent on constructing the SCHull graph.
![schull](https://github.com/Utah-Math-Data-Science/SCHull4Science/blob/main/SCHull_fig.png)

## Experiments
### Reaction Classification
```bash

python main_react.py --data_path <PATH_to_Data/Reaction-EC> --exp_name <Experiment_Name>

```

### Fold Classification
```bash

python main_fold.py --data_path <PATH_to_Data/FoldData> --exp_name <Experiment_Name>

```

### LBA
```bash

python main_lba.py --data_path <PATH_to_Data/LBA-split-by-sequence-identity-30> --exp_name <Experiment_Name>

```
