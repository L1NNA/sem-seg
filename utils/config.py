from typing import Optional, List
import torch


class Config:
    """
    A typing class for model configurations
    """

    def __init__(self):
        self.model:str
        self.model_id:str
        self.model_name:Optional[str]
        self.checkpoint:str
        self.log_dir:str
        self.seed:int
        self.comment:str

        self.training:bool
        self.validation:bool
        self.testing:bool
        
        self.data_path:str
        self.data:str
        self.database:Optional[str]
        self.vocab_size:int
        self.batch_size:int
        self.test_batch_size:int
        self.seq_len:int
        self.max_samples:int
        self.mem_len:int
        self.num_workers:int

        # labeling
        self.skip_label:int
        self.skip_seg:bool

        # transformer
        self.d_model:int
        self.n_heads:int
        self.n_layers:int
        self.d_ff:int
        self.dropout:float
        # cosformer
        self.cos_act:str
        # cats
        self.n_windows:int
        self.w_layers:int
        # auto bert
        self.bert_name:Optional[str]
        # COE
        self.sim_loss:str
        self.seg_masking:bool

        # self.itr:int
        self.epochs:int
        self.optim:str
        self.lr:float
        self.lr_warmup:int
        self.lr_decay:bool

        self.gpu:bool
        self.device:torch.device
        self.world_size:int
        self.rank:int
        self.local_rank:int
        self.is_host:bool
        self.distributed:bool
