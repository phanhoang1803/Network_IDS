import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def process_data(df):
    """
    This processing function is based on: https://www.kaggle.com/code/carlkirstein/unsw-nb15-modelling-97-7/notebook
    """
    list_drop = ['id', 'attack_cat']
    df.drop(list_drop, axis=1, inplace=True)
    
    num_cols = df.select_dtypes(exclude=['object']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    for col in num_cols:
        # Apply clamping to numerical columns
        # The features with a maximum value more than ten times median value is pruned to the 95th percentile
        max_val = df[col].max()
        if max_val > 10 * df[col].median() and max_val > 10:
            df[col] = df[col].clip(0, df[col].quantile(0.95))

        # Apply log function to nearly all numeric
        if df[col].nunique() > 50:
            # df[col] = np.log(df[col] + 1e-10)
            if df[col].min() == 0:
                df[col] = np.log(df[col] + 1)
            else:
                df[col] = np.log(df[col])
    
    # Reduce the labels in the categorical columns
    # Take the top 5 occuring labels in the feature as the labels and set the remainder to '-'
    for col in cat_cols:
        if df[col].nunique() > 6:
            df[col] = np.where(df[col].isin(df[col].value_counts().head(5).index), df[col], '-')
    
    # One-hot encoding
    encoder = OneHotEncoder()
    encoded = encoder.fit_transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out(cat_cols))
    df = pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)
    
    return df