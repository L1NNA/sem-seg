import glob
from os.path import join

import numpy as np
import torch
from tqdm import tqdm


def load_classification(files_path, range_len=100):
    seg_files = glob.glob(join(files_path, '*.pt'))

    training = []
    segments = []

    for seg_file in tqdm(seg_files):
        segs = torch.load(seg_file)

        # randomly pick a number from 1 to len(segs) - 1
        index = np.random.randint(1, len(segs) - 1)
        seg_ori, label = segs[index]
        prev_seg, prev_label = segs[index - 1]
        next_seg, next_label = segs[index + 1]

        seg = seg_ori
        prev_index = np.random.randint(-range_len, range_len)
        if prev_index < 0:
            seg = prev_seg[prev_index:] + seg
        else:
            seg = seg[prev_index:]

        post_index = np.random.randint(-range_len, range_len)
        if post_index < 0:
            seg = seg[:post_index]
        else:
            seg += next_seg[:post_index]

        training.append((seg, label))
        segments.append((seg_ori, label))
        segments.append((prev_seg, prev_label))
        segments.append((next_seg, next_label))

    return training, segments
