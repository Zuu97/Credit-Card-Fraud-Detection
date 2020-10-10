import numpy as np
import pandas as pd
from variables import*
# from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
from collections import Counter

def get_data():
    df = pd.read_csv(data_path)
    df_cols = df.columns.values[1:]

    Y = df[df_cols[-1]].astype(np.int32)
    X = df[df_cols[:-2]]

    return X, Y