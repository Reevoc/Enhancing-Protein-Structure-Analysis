import data_manipulation
import split
import normalization
from keras.utils import to_categorical
import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
import neural_networks
import sys
import pandas as pd


def parser():
    parser = argparse.ArgumentParser(
        description="prediction of the interactions for RING software"
    )
    parser.add_argument("-m", "--model", help="model name", required=True)
    parser.add_argument(
        "-n", "--normalization", help="normalization name", required=True
    )
    parser.add_argument(
        "-d", "--manipulation", help="dataset manipulation type", required=True
    )

    args = parser.parse_args()

    if not args.model:
        print("Error: Model name is required.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    if not args.normalization:
        print("Error: Normalization name is required.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    if not args.manipulation:
        print("Error: Dataset manipulation type is required.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    return args


if __name__ == "__main__":
    args = parser()

    if args.manipulation == "eliminate_unclassified":
        df = data_manipulation.generate_data_2()
        # df = pd.read_csv("./merged.tsv", sep="\t")

    if args.manipulation == "unclassified":
        df = data_manipulation.generate_data()
        # df = pd.read_csv("./merged.tsv", sep="\t")

    if args.normalization == "MinMaxScaler":
        df = normalization.all_normalization(df, "MinMaxScaler")
    if args.normalization == "StandardScaler":
        df = normalization.all_normalization(df, "StandardScaler")
    if args.normalization == "no_normalization":
        df = normalization.all_normalization(df, "no_normalization")

    if args.model == "model_1":
        loss, accuracy = neural_networks.model(df, neural_networks.create_model_1)
    if args.model == "model_2":
        loss, accuracy = neural_networks.model(df, neural_networks.create_model_2)
    if args.model == "model_3":
        loss, accuracy = neural_networks.model(df, neural_networks.create_model_3)

    with open("./results.csv", "a") as f:
        f.write(
            f"manipulation: {args.manipulation}, model: {args.model}, normalization: {args.normalization}, loss: {loss}, accuracy: {accuracy}\n"
        )
