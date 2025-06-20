# SCHull in Scientific Applications
This repository contains the code used for the experiments presented in the paper [A Theoretically-Principled Sparse, Connected, and Rigid Graph Representation of Molecules](https://openreview.net/pdf?id=OIvg3MqWX2), selected for an [Oral presentation at ICLR 2025](https://iclr.cc/virtual/2025/oral/31862).
<p align="center">
<img src="https://github.com/Utah-Math-Data-Science/SCHull4Science/blob/main/SCHull_fig.png" alt="Description of image" width="70%">
</p>

## Environment
All experiments were conducted using Python 3.9.20 within a [Conda 24.11.3](https://anaconda.org/anaconda/conda/files?page=2&sort=ndownloads&sort_order=asc&type=conda&version=24.11.3) environment, with CUDA 12.4 support on NVIDIA RTX 3090 GPUs. All required packages are specified in the [`requirements.txt`](https://github.com/Utah-Math-Data-Science/SCHull4Science/blob/main/requirements.txt) file.
 
## Dataset
### 1) Dataset Files
We use the reaction and fold datasets from [DIG](https://github.com/divelab/DIG/tree/dig-stable/dig/threedgraph/dataset#ecdataset-and-folddatase) and and the LBA dataset from [atom3d](https://github.com/drorlab/atom3d). We unified the storage format by saving all datasets as `.mdb` files, which keeps the total storage of the datasets under 2GB. The datasets are loaded as `LMDBDataset` in [`atom3d`](https://github.com/drorlab/atom3d) package when runing the experiments. 

Please download the dataset from the provided in the [releases](https://github.com/Utah-Math-Data-Science/SCHull4Science/releases/tag/v1.0.0) or using
```
wget https://github.com/Utah-Math-Data-Science/SCHull4Science/releases/download/v1.0.0/Data.zip
```
and `unzip Data.zip` before running the experiments. 

The data folders are organized as follows:
```bash
├── Data
│      │           
│      ├── Reaction-EC
│      │      │
│      │      ├── train
│      │      │      ├── data.mdb
│      │      │      ├── lock.mdb
│      │      │      ...
│      │      ├── val 
│      │      ...
│      ├── FoldData
│      ...               
```

### 2) Data Pre-processing
When the code is run for the first time, it will automatically pre-process the data, which includes constructing the original node features and the [SCHull](https://openreview.net/pdf?id=OIvg3MqWX2) graph using 
```
import SCHull; schull = SCHull.SCHull()
schull.get_schull
```
Each pre-processing step takes less than 10 minutes to complete, with approximately 40% of the time spent on constructing the SCHull graph.


## Experiments
We follow the section 3.4 in [SCHull paper](https://openreview.net/pdf?id=OIvg3MqWX2) to integrate the SCHull graph into the baseline models. The project codes are organized as:
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

### 1) Reaction Classification
```bash

python main_react.py --data_path <PATH_to_Data/Reaction-EC> \
                     --save_dir <PATH_to_SAVE> \
                     --exp_name <Experiment_Name> \
                     --schull <True_for_Integrating_SCHull>

```

### 2) Fold Classification
```bash

python main_fold.py --data_path <PATH_to_Data/FoldData> \
                    --save_dir <PATH_to_SAVE> \
                    --exp_name <Experiment_Name> \
                    --schull <True_for_Integrating_SCHull>

```

### 3) LBA Prediction
```bash

python main_lba.py --data_path <PATH_to_Data/LBA-split-by-sequence-identity-30> \
                   --save_dir <PATH_to_SAVE> \
                   --exp_name <Experiment_Name> \
                   --schull <True_for_Integrating_SCHull>

```

## Citation
You're welcome to cite our paper — we appreciate your support!
```
@inproceedings{wang2025schull,
  title={A Theoretically-Principled Sparse, Connected, and Rigid Graph Representation of Molecules},
  author={Wang, Shih-Hsin and Huang, Yuhao and Baker, Justin and Sun, Yuan-En and Tang, Qi and Wang, Bao},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

## Claim
This repository incorporates components of code from the [DIG](https://github.com/divelab/DIG).
