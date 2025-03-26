import csv
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset



def to_lowercase_int(val):
    # If val is in uppercase range, convert it to lowercase range
    # e.g., 36 <= val <= 61 => shift by 26
    if 36 <= val <= 61:
        return val - 26
    return val


class HandGestureTransferDatasetExp(Dataset):
    """
    Reads CSV data with columns [LineNumber, Inner, Outer, Label].
    Normalizes the data, then creates windows of size 'div'.
    Optionally performs an 80/20 split per label if 'train' is True/False.
    """
    def __init__(
        self,
        file,
        div=None,
        train=None,
        label_dict=None,
        min_res=None,
        max_res=None,
    ):
        self.index, self.resistance = [], []
        self.train = train

        with open(file, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            tmp_res = []
            tmp_label = []
            for row in reader:
                # row[1], row[2], row[3] are Inner, Outer, Label
                tmp_res.append([float(row[1]), float(row[2])])
                lbl_val = to_lowercase_int(int(row[3]))
                tmp_label.append(lbl_val)

        resistance = np.array(tmp_res)
        label = np.array(tmp_label)
        print("Loaded CSV:", resistance.shape, label.shape)

        # Build label_dict if not provided
        if label_dict is None:
            label_dict = {}
            idx_val = 0
            for l in label:
                if l not in label_dict:
                    label_dict[l] = idx_val
                    idx_val += 1

        # Filter valid labels
        filtered_res = []
        filtered_label = []
        for i in range(len(label)):
            if label[i] in label_dict:
                filtered_res.append(resistance[i])
                filtered_label.append(label[i])

        resistance = np.array(filtered_res)
        label = np.array(filtered_label)

        # Convert to label indices
        index_label = np.array([label_dict[l] for l in label])
        self.resistance = [resistance]
        self.index = [index_label]

        # Min-max normalization
        if min_res is not None and max_res is not None:
            self.min_res = min_res
            self.max_res = max_res
        else:
            self.min_res = np.amin(np.concatenate(self.resistance, axis=0), axis=0)
            self.max_res = np.amax(np.concatenate(self.resistance, axis=0), axis=0)

        for i in range(len(self.resistance)):
            self.resistance[i] = (self.resistance[i] - self.min_res) / (
                self.max_res - self.min_res
            )

        self.label_dict = label_dict
        self.index2label = {v: k for k, v in label_dict.items()}
        self.divide_data(div)

    def divide_data(self, window_size):
        self.total_data = {"sensor": [], "cur_idx": []}

        if self.train is None:
            # No train/test split, just create windows
            for j in range(len(self.index)):
                resistance, idx = self.resistance[j], self.index[j]

                for i in range(len(idx) - window_size):
                    self.total_data["sensor"].append(resistance[i : i + window_size])
                    self.total_data["cur_idx"].append(idx[i : i + window_size])
        else:
            # Train or test => do 80/20 split per label
            for j in range(len(self.index)):
                resistance, idx = self.resistance[j], self.index[j]

                label_windows = {}
                for i in range(len(idx) - window_size):
                    lbl = idx[i + window_size - 1]
                    if lbl not in label_windows:
                        label_windows[lbl] = {"sensor": [], "cur_idx": []}
                    label_windows[lbl]["sensor"].append(resistance[i : i + window_size])
                    label_windows[lbl]["cur_idx"].append(idx[i : i + window_size])

                for lbl in label_windows:
                    data_len = len(label_windows[lbl]["sensor"])
                    split_pt = int(data_len * 0.8)
                    if self.train:
                        use_sensor = label_windows[lbl]["sensor"][:split_pt]
                        use_idx = label_windows[lbl]["cur_idx"][:split_pt]
                    else:
                        use_sensor = label_windows[lbl]["sensor"][split_pt:]
                        use_idx = label_windows[lbl]["cur_idx"][split_pt:]

                    self.total_data["sensor"].extend(use_sensor)
                    self.total_data["cur_idx"].extend(use_idx)

        # Convert to numpy arrays
        for key in self.total_data:
            self.total_data[key] = np.array(self.total_data[key])
            print(f"{key} shape: {self.total_data[key].shape}")

    def __len__(self):
        return len(self.total_data["sensor"])

    def __getitem__(self, idx):
        return {
            "input": torch.from_numpy(self.total_data["sensor"][idx]).float(),
            "label": torch.from_numpy(self.total_data["cur_idx"][idx]).long(),
            "window_idx": idx,
        }

