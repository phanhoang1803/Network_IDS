import pandas as pd
import numpy as np

from data.data_processing import *

def load_data(csv_path, CONFIG):
    df = pd.read_csv(csv_path)
    
    if CONFIG["debug"]:
        df = df[:10000]
    
    df = generate_features(df)
        
    return df