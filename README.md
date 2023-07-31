# SemSeg
Semantic Code Segmentation, paper [link](https://www.overleaf.com/project/6331cafdb13619ac9254afd8)

## Setup

Use conda to manage environments
```shell script
conda create -n sem_seg python=3.8
conda activate sem_seg
# Install pytorch
pip install numpy pytorch
```

Baselines & Tokenizer
```bash
pip install transformers
```


### TODO
- [x] Optimize data loader: more data and sync dataset
- [ ] predicter: add Tokenizer & predicter
- [ ] add logger