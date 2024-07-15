import numpy as np
import torch
from torch.utils.data import Dataset


class UNSW_NB15_Dataset(Dataset):
    def __init__(self, df, CONFIG=None):
        self.df = df
        
        drop_cols = ["label"]
        if "kfold" in df.columns:
            drop_cols.append("kfold")
            
        self.x_data = df.drop(columns=drop_cols).values.astype(np.float32)
        self.y_data = df["label"].values.astype(np.float32)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        return {
            "x": torch.from_numpy(self.x_data[index]),
            "y": torch.tensor(self.y_data[index], dtype=torch.float32)
        }
        
# class UNSW_NB15_Dataset(Dataset):
#     def __init__(self, df, CONFIG=None):
#         self.df = df
        
#     def __len__(self):
#         return len(self.df)
    
#     def __getitem__(self, index):
#         x = torch.tensor(self.df.drop(columns=["label", "kfold"]).iloc[index].values, dtype=torch.float)
#         y = torch.tensor(self.df["label"].iloc[index], dtype=torch.float)
        
#         return {
#             "x": x,
#             "y": y
#         }