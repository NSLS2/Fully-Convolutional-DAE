#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import date

class CorrDataSet(Dataset):
    """To use CorrDataSet:

    ds = CorrDataSet(data_file)
    dataloader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    for sample_batched in dataloader:
        do_something(sample)
    """

    def __init__(self, data_file):
        self.data_dict = torch.load(data_file)
        self.uids = list(self.data_dict.keys())
        np.random.shuffle(self.uids)

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):

        uid = self.uids[idx]
        raw_data_path = self.data_dict[uid]["data"]
        target_path = self.data_dict[uid]["target"]
        raw_data = torch.load(raw_data_path)

        target_data = torch.load(target_path)

        sample = {"data": raw_data, "target": target_data}

        return sample
