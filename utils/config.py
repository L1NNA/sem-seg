from typing import Optional, List


class Config:

    def __init__(self):
        pass

        self.training:bool
        self.model:str
        self.model_id:str
        self.des:str
        self.seed:int
        
        self.batch_size:int
        self.seq_len:int
        self.weights_path:str

        self.d_model:int
        self.n_heads:int
        self.e_layers:int
        self.d_layers:int
        self.d_ff:int
        self.dropout:float

        self.num_workers:int
        self.itr:int
        self.training_epochs:int
        self.patience:int
        self.learning_rate:float
        self.loss:str

        self.gpu:Optional[List[int]]
        self.device = None
