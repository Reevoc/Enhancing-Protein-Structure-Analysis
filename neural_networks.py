import data_manipulation
import split
import normalization
from keras.utils import to_categorical

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np


def creating_dataset_for_train(df):
    X, Y = split.split_dataset(df)

    X = X.astype(float)
    # print data types of Y
    print(Y.dtypes)
    Y = to_categorical(Y)
    # split the dataset into train and test
    X_train, X_test, Y_train, Y_test = split.split_train_test(X, Y)
    print("dataset splitted...")

    print(
        f"information about len of the dataset: {X_train.shape, Y_train.shape, X_test.shape, Y_test.shape}"
    )
    return X_train, X_test, Y_train, Y_test


def create_input_dim(X_train, Y_train):
    input_dim = X_train.shape[1]
    num_classes = Y_train.shape[1]
    return input_dim, num_classes


def create_model(input_dim, num_classes):
    model = keras.Sequential()
    model.add(layers.Dense(128, activation="relu", input_dim=input_dim))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(num_classes, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def create_model_2(input_dim, num_classes):
    model = keras.Sequential()


def evaluate_model(model, X_test, Y_test):
    loss, accuracy = model.evaluate(X_test, Y_test)
    print("Accuracy", accuracy)


def model_1(df):
    X_train, X_test, Y_train, Y_test = creating_dataset_for_train(df)
    input_dim, num_classes = create_input_dim(X_train, Y_train)
    model = create_model(input_dim, num_classes)
    model.fit(
        X_train,
        Y_train,
        epochs=10,
        batch_size=128,
        verbose=True,
        validation_data=(X_test, Y_test),
    )
    evaluate_model(model, X_test, Y_test)


def model_2(df):
    X_train, X_test, Y_train, Y_test = creating_dataset_for_train(df)
    input_dim, num_classes = create_input_dim(X_train, Y_train)
    model = create_model(input_dim, num_classes)
    class_weight = {}
    Y_train = Y_train.astype("int64")

    for label in range(num_classes):
        label_counts = np.bincount(Y_train[:, label])
        print(label_counts)
        print(label_counts.max() / label_counts[label])
        class_weight[label] = label_counts.max() / label_counts[label]

    model.fit(
        X_train,
        Y_train,
        epochs=10,
        batch_size=128,
        verbose=True,
    )

    evaluate_model(model, X_test, Y_test)
