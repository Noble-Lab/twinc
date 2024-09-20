# TwinC

# What is TwinC?
TwinC is a Python package for training, inference, and interpretation of trans-3D genome folding in humans. TwinC uses a Convolutional Neural Network model that predicts contact between two _trans_ genomic loci from nucleotide sequences. The model takes two 100 kbp nucleotide sequences as input and treats the task of predicting _trans_ contacts as a classification task.

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

# Cite

TODO

# Contributing

TODO