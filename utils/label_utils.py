from typing import List, Tuple
import re
from enum import Enum
from collections import Counter


class SegType(Enum):
    UNK = 0
    BOOTSTRAP = 1
    LOCAL = 2
    NODE_MODULE = 3

    @classmethod
    def max_value(self, seg_type)->bool:
        return seg_type == max(SegType, key=lambda c: c.value)

    @classmethod
    def min_value(self) -> int:
        return min(SegType, key=lambda c: c.value).value


NODE_MODULES = '/node_modules/'
BOOTSTRAPS = (
    'webpack/universalModuleDefinition',
    'webpack/bootstrap',
    'webpack/startup',
    '/runtime/[a-zA-Z]+',
    '/external',
)

UNK = '<UNK>'


def get_seg_type(label:str) -> SegType:
    """
    Map the source name to a segmentation type
    """
    if label is None:
        return SegType.UNK

    # The segment refers to a node module package
    if label.find(NODE_MODULES) > -1:
        return SegType.NODE_MODULE
    
    # label boostrap segments
    blabel = label.split(' ')[0]
    blabel = blabel.split('?')[0]

    for boostrap in BOOTSTRAPS:
        if re.match('^webpack://.*' + boostrap + '$', blabel):
            return SegType.BOOTSTRAP

    if label == UNK:
        return SegType.UNK
    
    # Local files
    if label.find('./') > -1:
        return SegType.LOCAL

    return SegType.UNK

def get_simple_label(label:str) -> SegType:
    if label is None:
        return SegType.BOOTSTRAP
    for boostrap in BOOTSTRAPS:
        if re.match('^webpack://.*' + boostrap + '.*', label):
            return SegType.BOOTSTRAP
    return SegType.LOCAL



def label_seg(labels:List[str]) -> int:
    # curr_value = SegType.min_value()
    
    # for label in labels:
    #     seg_type = get_seg_type(label)
    #     if SegType.max_value(seg_type):
    #         return seg_type.value
    #     if seg_type.value > curr_value:
    #         curr_value = seg_type.value
    # return curr_value
    values = {}
    max_value, max_type = 0, None
    for label in labels:
        if label in values:
            values[label] += 1
        else:
            values[label] = 1
        if values[label] > max_value:
            max_type = get_seg_type(label).value
            max_value = values[label]
    return max_type

def get_most_common_label(labels:List[str]) -> str:
    # Create a Counter object
    counter = Counter(labels)

    # Get the most common string(s)
    most_common_strings = counter.most_common(1)

    return most_common_strings[0][0]

def get_num_of_labels(config):
    return len(SegType) - config.skip_label

