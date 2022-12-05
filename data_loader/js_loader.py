from os import listdir
from os.path import isfile, join
import random

import torch
from torch.utils.data import Dataset


class SimpleSegmentaiton(Dataset):

    def __init__(self, dest_path, labels=10):
        js_files = [f for f in listdir(dest_path) if isfile(join(dest_path, f)) and f.endswith('.pt')]
        random.shuffle(js_files)

        file_mappings = {}
        selected_files = []
        each_label = labels / 2
        for f in js_files:
            base = int(f.split('_')[0])
            label = int(f.split('_')[2])
            if base in file_mappings:
                if file_mappings[base][label] < each_label:
                    file_mappings[base][label] += 1
                    selected_files.append(f)
            else:
                file_mappings[base] = [0, 0]
                file_mappings[base][label] += 1
                selected_files.append(f)

        self.selected_files = selected_files
        self.dest_path = dest_path

    def __len__(self):
        return len(self.selected_files)

    def __getitem__(self, index):
        f = self.selected_files[index]
        return torch.load(join(self.dest_path, f))
