import argparse
import datetime
import sys

from keras.models import load_model
from keras.optimizers import Adam

import configuration as conf
import src.neural_networks as neural_networks
from src.manage_data import build_index, check_resi, prepare_data
from src.normalization import normalization_df


def parser():
    parser = argparse.ArgumentParser(
        description="prediction of the interactions for RING software"
    )
    parser.add_argument("-m", "--model", help="model name", required=False)
    parser.add_argument(
        "-n", "--normalization", help="normalization name", required=False
    )
    parser.add_argument(
        "-d", "--manipulation", help="dataset manipulation type", required=False
    )

    args = parser.parse_args()

    if len(sys.argv) <= 1:
        return []

    error = []
    if not args.model:
        error.append("Model name is required.")

    if not args.normalization:
        error.append("Normalization name is required.")

    if not args.manipulation:
        args.manipulation = False

    if error:
        parser.print_help()
        raise SystemExit(f"\n{' '.join(error)}")

    return args


def main(df, model_name, normalization_mode, remove_unclassified, f):
    print(model_name, normalization_mode)
    df = normalization_df(df, normalization_mode)
    neural_networks.kfold_train(df, model_name, f)


if __name__ == "__main__":
    args = parser()
    now = datetime.datetime.now().strftime(r"%m-%d_%H:%M")
    out = f"./results{now}.csv"

    conf.DEBUG = True

    index = build_index(
        path_pdb=conf.PATH_PDB,
        path_ring=conf.PATH_FEATURES_RING,
        path_tsv=conf.PATH_DSSP_TSV,
        generate_new=False,
        update_dssp=False,
    )

    df = prepare_data(index)
    df.to_csv("data.tsv")

    with open(out, "w") as f:
        header = f"model_name\taverage_accuracy\taverage_f1\taverage_precision\taverage_recall\n"
        f.write(header)

        if len(sys.argv) > 1:
            main(df, args.model, args.normalization, args.manipulation, f)
        else:
            ## gridsearch
            scales = ["StandardScaler"]
            models = ["model_2", "model_3"]
            optimizers = [Adam(learning_rate=0.001)]
            dropout_rate = [0.2]
            epochs = [20]
            conf.KFOLDS = 2
            params = [optimizers, dropout_rate, epochs]
            for scale in scales:
                f.write(f"\n### {scale}\n")
                df_norm = normalization_df(df, scale)
                for model_name in models:
                    print(f"Start training {model_name}\n")
                    # neural_networks.kfold_train(df_norm,model_name,f)
                    neural_networks.gridsearch(df_norm, model_name, f, params)

            ## Generate Model
            # df_norm = normalization_df(df, "StandardScaler")
            # neural_networks.train(df_norm, "model_3")

            ## Predict
            # # df_norm = normalization_df(df, "StandardScaler")
            # model = load_model("model_3.h5")
            # neural_networks.predict(df_norm, model)
