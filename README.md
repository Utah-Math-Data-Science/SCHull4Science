# SCHull for Protein
Code for protein experiments in [SCHull](https://openreview.net/pdf?id=OIvg3MqWX2).
## Environment
All experiments were implemented using Python 3.9.20 and Conda 24.11.3, with the required packages listed in the `requirements.txt` file.
 
## Dataset
### Dataset Files
All datasets are saved as `.mdb` files. Please download the dataset from the provided link and unzip it before running the experiments. The data folders are organized as follows:
```bash
├── Data
│     │           
│     ├── Reaction-EC
│     │          ├── data.mdb
│     │          ├── lock.mdb
│     │          ├── ...
│     ├── FoldData
│     │          ...
│     │                    
│   ...
```
### Data Pre-processing
When the code is run for the first time, it will automatically pre-process the data, which includes constructing the original node features and the [SCHull](https://openreview.net/pdf?id=OIvg3MqWX2) graph. Each pre-processing step takes less than 10 minutes to complete, with approximately 40% of the time spent on constructing the SCHull graph.
