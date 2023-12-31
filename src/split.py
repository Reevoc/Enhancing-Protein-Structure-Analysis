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
        n = min(target_samples, subset.shape[0])
        print(f"Sampling {n} from {interaction_type}")
        sampled_subsets[interaction_type] = subset.sample(
            n=n, random_state=42, replace=True
        )

    uniform_training_set = pd.concat(sampled_subsets.values())
    uniform_training_set = uniform_training_set.sample(frac=1, random_state=42)

    return uniform_training_set


def drop_useless_column(df):
    print("Drop useless columns")
    col_names = ["pdb_id", "s_ch", "s_resi", "s_ins", "t_ch", "t_resi", "t_ins"]
    df.drop(col_names, axis=1, inplace=True)
    return df


def create_X(df, inference):
    column2drop = [
        "pdb_id",
        "s_ch",
        "s_resi",
        "s_ins",
        "t_ch",
        "t_resi",
        "t_ins",
        "Interaction",
        "OrgInteraction",
    ]
    if inference:
        X = df.drop(column2drop[:-2], axis=1).copy()
    else:
        X = df.drop(column2drop, axis=1).copy()
    return X.values


def create_Y(df):
    Y = to_categorical(df["Interaction"])
    # Y = np.array([[i for i in j] for j in Y])
    # write in label.csv Y and df["OrgInteraction"]
    with open(f"label.csv", "w") as ff:
        ff.write("Y\tOrgInteraction\n")
        for i, j in zip(map(np.argmax, Y), df["OrgInteraction"]):
            ff.write(f"{i}\t{j}\n")
    return Y


def get_dataset(
    df, balanced=False, inference=False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if balanced:
        df = balance_interaction_types(df)
    X = create_X(df, inference)
    if inference:
        return X

    Y = create_Y(df)
    return X, Y
