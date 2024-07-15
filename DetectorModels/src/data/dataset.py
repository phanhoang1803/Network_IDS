import torch
from torch.utils.data import Dataset


class UNSW_NB15_Dataset(Dataset):
    def __init__(self, df, CONFIG=None):
        self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        x = torch.tensor(self.df.drop(columns=["label", "kfold"]).iloc[index].values, dtype=torch.float)
        y = torch.tensor(self.df["label"].iloc[index], dtype=torch.float)
        
        return {
            "x": x,
            "y": y
        }