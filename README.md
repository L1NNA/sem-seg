# SBoM
Software Bill-of-Materials by Chain-of-Experts

## Setup

Use conda to manage environments
```shell script
conda create -n sem_seg python=3.8
conda activate sem_seg
# Install pytorch with cuda
pip install numpy scikit-learn # tools
pip install transformers
pip install tensorboard
```
<!-- pip install hnswlib # indexing -->

## Data Preparation
Download the [data.tar.gz](https://github.com/L1NNA/sem-seg/releases/download/v1.0.0/data.tar.gz) from release page and extract the tarball as following
```
mkdir data
tar -xzvf data.tar.gz -C data
```
The data has already been separated into train, validation, and test.
Each file has already been preprocessed into the following format `List[(segment:str, label_name:str, source_string:str)]`

## Running the model
Check `experiments` folder for scripts to train/test the model.
