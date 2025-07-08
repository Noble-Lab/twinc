# Full dataset

This dataset can be used to reproduce results in the [TwinC manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC11429679/). Please download this data in the same folder as the [TwinC paper repository](https://github.com/Noble-Lab/twinc_paper). We have submitted our training datasets to [zenodo.org/records/15802811](https://zenodo.org/records/15802811). The datasets are divided into three parts:

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

This dataset contains: 

1. **hg38.no_Y_MT.fa**: Fasta file for the human genome containing chromosomes 1-22 and X.
   
2. **hg38.no_Y_MT.fa.fai**: Index for the fasta file.
   
3. **hg38.no_Y_MT.memmap**: One-hot-encoded memory map for the human genome.
   
```
A -> [1, 0, 0, 0]
C -> [0, 1, 0, 0]
G -> [0, 0, 1, 0]
T -> [0, 0, 0, 1]
```

4. **gene_tpm_2017-06-05_v8_heart_left_ventricle.gct**: TPMs for genes in the heart's left ventricle from the GTEX. 

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