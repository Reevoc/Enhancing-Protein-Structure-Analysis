from typing import Tuple

import numpy as np
import pandas as pd
from keras.utils import to_categorical


def balance_interaction_types(df):
    interaction_counts = df["Interaction"].value_counts()

    interaction_subsets = {}
    for interaction_type, count in interaction_counts.items():
        interaction_subsets[interaction_type] = df[
            df["Interaction"] == interaction_type
        ]

    target_samples = int(min(interaction_counts.values) * 4)
    sampled_subsets = {}
    for interaction_type, subset in interaction_subsets.items():
        sampled_subsets[interaction_type] = subset.sample(
            n=target_samples, random_state=42
        )

    uniform_training_set = pd.concat(sampled_subsets.values())
    uniform_training_set = uniform_training_set.sample(frac=1, random_state=42)


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


def get_dataset(df, balanced=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if balanced:
        df = balance_interaction_types(df)
    X = create_X(df)
    Y = to_categorical(df["Interaction"])

    return X, Y
