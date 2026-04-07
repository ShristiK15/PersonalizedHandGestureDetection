import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os

class JesterSkeletonDataset(Dataset):

    def __init__(self, csv_file, data_dir):
        self.data_dir = data_dir
        
        raw_df = pd.read_csv(csv_file)
        
        # 1. Prevent the Ghost File Crash
        valid_rows = []
        for idx, row in raw_df.iterrows():
            path = os.path.join(self.data_dir, f"{row['video_id']}.npy")
            if os.path.exists(path):
                valid_rows.append(row)
                
        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"Loaded {len(self.df)} valid sequences from {csv_file}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        video_id = self.df.iloc[idx]["video_id"]
        # Now we can just pull the integer directly from your CSV
        label = self.df.iloc[idx]["label"]

        path = os.path.join(self.data_dir, f"{video_id}.npy")
        
        sequence = np.load(path)
        sequence = torch.tensor(sequence, dtype=torch.float32)

        # Cast directly to LongTensor for PyTorch
        label_tensor = torch.tensor(label, dtype=torch.long)

        return sequence, label_tensor