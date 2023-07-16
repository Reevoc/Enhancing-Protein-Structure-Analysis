import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def drop_useless_column(df):
    col_names = ["pdb_id", "s_ch", "s_resi", "s_ins", "t_ch", "t_resi", "t_ins"]
    df.drop(col_names, axis=1, inplace=True)
    print(df.head(5).T)
    return df


def transform_alphabetical_numerical(df):
    columns = ["s_resn", "s_ss8", "s_ss3", "t_resn", "t_ss8", "t_ss3"]
    label_encoder = preprocessing.LabelEncoder()
    for col in columns:
        df[col] = label_encoder.fit_transform(df[col])
    print(df.head(5).T)
    return df


def create_X(df):
    X = df.drop(["Interaction"], axis=1).copy()
    return X


def create_Y(df):
    Y = df["Interaction"]
    return Y


def split_train_test(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1, shuffle=True
    )
    return X_train, X_test, Y_train, Y_test


def split_dataset(df):
    df = drop_useless_column(df)
    df = transform_alphabetical_numerical(df)
    X = create_X(df)
    Y = create_Y(df)
    return X, Y
