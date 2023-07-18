import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def drop_useless_column(df):
    col_names = ["pdb_id", "s_resi", "s_ins", "t_resi", "t_ins"]
    df.drop(col_names, axis=1, inplace=True)
    print(df.head(5).T)
    return df


def create_X(df):
    X = df.drop(["Interaction"], axis=1).copy()
    return X


def create_Y(df):
    df["Interaction"] = df["Interaction"].astype(int)
    Y = df["Interaction"]
    print(Y)
    return Y


def split_train_test(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, shuffle=True
    )
    return X_train, X_test, Y_train, Y_test


def split_dataset(df):
    df = drop_useless_column(df)
    X = create_X(df)
    Y = create_Y(df)
    return X, Y
