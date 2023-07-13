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
        
        self.batch_size:int
        self.seq_len:int
        self.mem_len:int
        self.data_path:str
        self.data:str

        # transformer
        self.d_model:int
        self.n_heads:int
        self.n_layers:int
        self.d_ff:int
        self.dropout:float
        # cosformer
        self.cos_act:str
        # expire span
        self.expire_span_alpha:float
        self.expire_span_ramp:int
        self.expire_span_pre_div:float

        self.itr:int
        self.epochs:int
        self.lr:float
        self.optim:str

        self.gpu:Optional[List[int]]
        self.device:torch.device
        self.rank:int
        self.world_size:int
        self.local_rank:int
