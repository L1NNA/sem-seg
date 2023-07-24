from typing import Optional, List
import torch


class Config:

    def __init__(self):
        pass

        self.training:bool
        self.model:str
        self.model_name:str
        self.checkpoint:str
        self.seed:int
        self.checkpoint:str
        
        self.data_path:str
        self.data:str
        self.vocab_size:int
        self.batch_size:int
        self.seq_len:int
        self.mem_len:int
        self.num_workers:int

        # transformer
        self.d_model:int
        self.n_heads:int
        self.n_layers:int
        self.d_ff:int
        self.dropout:float
        # cosformer
        self.cos_act:str
        # cats
        self.w_layers:int
        self.coherence_hinge_margin:float

        # self.itr:int
        self.epochs:int
        self.optim:str
        self.lr:float
        self.lr_warmup:int
        self.lr_decay:bool

        self.gpu:Optional[List[int]]
        self.device:torch.device
        self.world_size:int
        self.rank:int
