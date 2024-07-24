import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

def generate_features(df):
    # Duration
    # df['duration'] = df['Ltime'] - df['Stime']
    
    # Ratios
    df['byte_ratio'] = df['sbytes'] / (df['dbytes'] + 1)
    df['tcp_setup_ratio'] = df['tcprtt'] / (df['synack'] + df['ackdat'] + 1)
    
    # Aggregate Features
    df['total_bytes'] = df['sbytes'] + df['dbytes']
    df['total_tcp_setup'] = df['tcprtt'] + df['synack'] + df['ackdat']
    
    # Interaction Features
    
    # Statistical Features
    df['tcp_seq_diff'] = df['stcpb'] - df['dtcpb']
    
    return df

def process_data(df, encoder=None, scaler=None):
    """
    This processing function is based on: https://www.kaggle.com/code/carlkirstein/unsw-nb15-modelling-97-7/notebook
    """
    df.drop(['id', 'attack_cat'], axis=1, inplace=True)
    
    num_cols = df.select_dtypes(exclude=['object']).columns.drop('label')
    cat_cols = df.select_dtypes(include=['object']).columns
    
    df[num_cols] = df[num_cols].apply(lambda x: x.clip(0, x.quantile(0.95)) if x.max() > 10 * x.median() and x.max() > 10 else x)
    df[num_cols] = df[num_cols].apply(lambda x: np.log(x + 1) if x.min() == 0 else np.log(x) if x.nunique() > 50 else x)
    
    df[cat_cols] = df[cat_cols].apply(lambda x: x.where(x.isin(x.value_counts().head(5).index), '-') if x.nunique() > 6 else x)
    
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(df[cat_cols])
    else:
        encoded = encoder.transform(df[cat_cols])
    
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
    
    if scaler is None:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])
    
    df = pd.concat([df["label"], df[num_cols], encoded_df], axis=1)
    
    return df, encoder, scaler