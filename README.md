# SemSeg
Semantic Code Segmentation, paper [link](https://www.overleaf.com/project/6331cafdb13619ac9254afd8)

## Setup

Use conda to manage environments
```shell script
conda create -n sem_seg python=3.8
conda activate sem_seg
# Install pytorch with cuda
pip install numpy scikit-learn # tools
pip install transformers
pip install hnswlib # indexing
```

### TODO
- [x] Optimize data loader: more data and sync dataset
- [x] predicter: add Tokenizer & predicter
- [ ] add logger