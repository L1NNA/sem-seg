import torch.distributed as dist

# from data_loader.segmentation_dataset import SegmentationDataset
# from data_loader.labeling_dataset import LabelingDataset
from data_loader.coe_dataset import COEDataset
# from data_loader.multi_window_segmentation_dataset import MultiWindowSegmentationDataset
# from data_loader.siamese_clone_dataset import SiameseCloneDataset
# from data_loader.single_labeling_dataset import SingleLabelingDataset
# from data_loader.coes_dataset import CoEsDataset
from utils.setup_BPE import get_tokenizer
from utils.dist_dataset import DistDataset
from utils.config import Config
from utils.label_utils import get_num_of_labels


DATASET_MAP = {
    'segmentation': (SegmentationDataset, lambda _: 2),
    'labeling': (LabelingDataset, lambda config: get_num_of_labels(config)),
    'coe': (CoEsDataset, lambda config:get_num_of_labels(config)), # COEDataset
    'siamese_clone': (SiameseCloneDataset, lambda _:0),
    'multi_window_seg': (MultiWindowSegmentationDataset, lambda _:2),
    'single_labeling': (SingleLabelingDataset, lambda config:get_num_of_labels(config)),
}


def load_dataset(config:Config):
    clazz, get_output_dim = DATASET_MAP[config.data]
    train, valid, test = _load_all(clazz, config)
    _load_cache(train, config)
    _load_cache(valid, config)
    _load_cache(test, config)
    return train, valid, test, get_output_dim(config)

def _load_all(clazz, config:Config):
    train, valid, test = None, None, None
    if config.training:
        train = clazz(config, 'train')
        valid = clazz(config, 'valid')
    if config.validation and valid is None:
        valid = clazz(config, 'valid')
    if config.testing:
        test = clazz(config, 'test')
    return train, valid, test

def _load_cache(dataset:DistDataset, config:Config):
    if dataset is None:
        return
    dataset.pipeline()

def load_tokenizer(config:Config):
    tokenizer = get_tokenizer(config.bert_name)
    config.vocab_size = tokenizer.vocab_size