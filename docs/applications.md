## Full dataset

This dataset can be used to reproduce results in the TwinC manuscript. Please download this data in the same folder as the [TwinC paper repository](https://github.com/Noble-Lab/twinc_paper). We have submitted our training datasets to [zenodo.org/records/15802811](https://zenodo.org/records/15802811). The datasets are divided into three parts:

### Data
The sequence data can be downloaded using:

```
cd twinc-paper/
wget https://zenodo.org/records/15802811/files/TwinC_data_resources_V1.tar.gz
```

After downloading, you can uncompress the file, and it will populate the data folder. 

```
tar -xvzf TwinC_data_resources_V1.tar.gz 
```

### Labels
Labels for training, as well as intermediate results for reproducing the figures, can be downloaded using:
```
cd twinc-paper/
wget https://zenodo.org/records/15802811/files/TwinC_label_resources_V1.tar.gz
```
After downloading, you can uncompress the file, and it will populate the results folder. 

```
tar -xvzf TwinC_label_resources_V1.tar.gz 
```

### Models
You can download the pretrained models using:
```
cd twinc-paper/
wget https://zenodo.org/records/15802811/files/TwinC_models_V1.tar.gz
```
After downloading, you can uncompress the file, and it will populate the models folder. 

```
tar -xvzf TwinC_models_V1.tar.gz 
```
