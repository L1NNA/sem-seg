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


### Ideas v - Yes, x - No 

- [ ] CoE
    - [x] Coefficients for Losses: plays no role
    - [v] Shared backbone has no impact on performance compared to standlone models
    - [x] FFN output for n classes - not relevant
    - [v] More windows?
- Seg:
    - [v] longer or shorter?
    - [x] Multiple windows: slow, same performance
- Cls
    - [v] longer sequences are generally better
    - [x] segment windows has low performance
- Ccr
    - [x] Longformer
    - [x] Gzip()