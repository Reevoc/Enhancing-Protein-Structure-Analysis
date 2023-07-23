from typing import Tuple

import numpy as np
import pandas as pd
from keras.utils import to_categorical


def drop_useless_column(df):
    print("Drop useless columns")
    col_names = ["pdb_id", "s_ch", "s_resi", "s_ins", "t_ch", "t_resi", "t_ins"]
    df.drop(col_names, axis=1, inplace=True)
    return df


def create_X(df):
    X = df.drop(
        ["pdb_id", "s_ch", "s_resi", "s_ins", "t_ch", "t_resi", "t_ins", "Interaction"],
        axis=1,
    ).copy()
    return X.values


def create_Y(df):
    Y = df["Interaction"]
    Y = np.array([[i for i in j] for j in Y])
    return Y


def get_dataset(df) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = create_X(df)
    Y = to_categorical(df["Interaction"])

    return X, Y
