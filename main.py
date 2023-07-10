import data_manipulation
import split
import normalization
from keras.utils import to_categorical
import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MultiLabelBinarizer
import neural_networks
import sys


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

    print("start manipulation...")
    if args.manipulation == "unclassified":
        df = data_manipulation.generate_interaction_dataframe(
            data_manipulation.process_files_unclassified
        )
    if args.manipulation == "eliminate_unclassified":
        df = data_manipulation.generate_interaction_dataframe(
            data_manipulation.process_files_eliminate_unclassified
        )

    print("start normalization...")
    if args.normalization == "normalization_all":
        df = normalization.all_normalization(df)

    print("start evaluation...")
    if args.model == "model_1":
        neural_networks.model_1(df)
    if args.model == "model_2":
        neural_networks.model_2(df)
