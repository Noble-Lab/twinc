# Downloads

## Example dataset
We are providing an example dataset to test your TwinC installation and to prepare data, label and configuration files for training TwinC in a new tissue or cell line.  

```
cd twinc/
wget https://zenodo.org/records/15802811/files/TwinC_example_V1.tar.gz
```

After downloading, you can uncompress the file, and it will populate the data folder. 

```
tar -xvzf TwinC_example_V1.tar.gz 
```

This dataset contains: 
1. hg38.no_Y_MT.fa: Fasta file for the human genome containing chromosomes 1-22 and X.
   
2. hg38.no_Y_MT.fa.fai: Index for the fasta file.
   
3. hg38.no_Y_MT.memmap: One-hot-encoded memory map for the human genome.
   ```
   A -> [1, 0, 0, 0]
   C -> [0, 1, 0, 0]
   G -> [0, 0, 1, 0]
   T -> [0, 0, 0, 1]
   ```
   
4. train_labels.txt: Training labels. A text file containing genomic coordinates from chromosomes A and B, a contact label (0 -> no contact, 1 -> contact) and KR-normalized contact frequency. The columns in the files are as follows:
   ```
   1. chrA_Name: Name of the first chromosome.
   2. chrA_Start: Start coordinate in chrA.
   3. chrA_End: End coordinate in chrA.
   4. chrB_Name: Name of the second chromosome.
   5. chrB_Start: Start coordinate in chrB.
   6. chrB_End: End coordinate in chrB.
   7. Contact_Label: Whether the two loci are in contact (0->No, 1->Yes).
   8. Contact_Frequency: KR-Normalized contact frequency.
   ```
   
5. val_labels.txt: Validation labels.

6. test_labels.txt: Test labels.  



