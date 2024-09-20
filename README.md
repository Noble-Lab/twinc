# TwinC

# What is TwinC?
TwinC is a Python package for training, inference, and interpretation of trans-3D genome folding in humans. TwinC uses a Convolutional Neural Network model that predicts contact between two _trans_ genomic loci from nucleotide sequences. The model takes two 100 kbp nucleotide sequences as input and treats the task of predicting _trans_ contacts as a classification task. To reproduce analyses performed in the **Jha et al., 2024**, please check out [TwinC 2024 manuscript repository](https://github.com/Noble-Lab/twinc_manuscript).

# Installation
```
python -m pip install git+https://github.com/Noble-Lab/twinc
```

# Usage

Training a new TwinC model
```
twinc_train --config_file configs/heart_left_ventricle.yml 
```

Inference on TwinC model
```
twinc_test --config_file configs/heart_left_ventricle.yml 
```

# Citation

**Jha A, Hristov B, Wang X, Wang S, Greenleaf WJ, Kundaje A, Aiden EL, Bertero A, Noble WS.**  Prediction and functional interpretation of inter-chromosomal genome architecture from DNA sequence with TwinC. *bioRxiv* (2024): 2024-09.


# Contributions

We welcome any bug reports, feature requests or other contributions. Please submit a well-documented report on our [issue tracker](https://github.com/Noble-Lab/twinc/issues). For substantial changes, please fork this repo and submit a pull request for review.

See [CONTRIBUTING.md](/CONTRIBUTING.md) for additional details.

You can find [official releases](https://github.com/Noble-Lab/twinc/releases) here.