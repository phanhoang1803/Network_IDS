import pandas as pd
import numpy as np

from data.data_processing import *

def load_data(csv_path, CONFIG):
    df = pd.read_csv(csv_path)
    
    df = process_data(df)
    
    return df