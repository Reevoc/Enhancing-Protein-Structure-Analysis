import argparse
import datetime
import sys

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
    # parser.add_argument(
    #     "-d", "--manipulation", help="dataset manipulation type", required=False
    # )

    args = parser.parse_args()

    if len(sys.argv) <= 1:
        return []

    error = []
    if not args.model:
        error.append("Model name is required.")

    if not args.normalization:
        error.append("Normalization name is required.")

    # if not args.manipulation:
    #     error.append("Dataset manipulation type is required.")

    if error:
        parser.print_help()
        raise SystemExit(f"\n{' '.join(error)}")

    return args


def main(df, model_name, normalization_mode, f):
    print(model_name, normalization_mode)
    df = normalization_df(df, normalization_mode)
    neural_networks.kfold_train(df, model_name, f)


if __name__ == "__main__":
    args = parser()
    now = datetime.datetime.now().strftime(r"%m-%d_%H:%M")
    out = f"./results{now}.csv"

    conf.DEBUG = True
    # conf.MTHREAD=1

    index = build_index(
        path_pdb=conf.PATH_PDB,
        path_ring=conf.PATH_FEATURES_RING,
        path_tsv=conf.PATH_DSSP_TSV,
        generate_new=False,
        update_dssp=False,
    )

    df = prepare_data(index)
    # df.to_csv("data.tsv")

    with open(out, "w") as f:
        f.write(
            f"model_name\taverage_accuracy\taverage_f1\taverage_precision\taverage_recall\n"
        )
        if len(sys.argv) > 1:
            main(df, args.model, args.normalization, f)
            # scale = args.normalization
            # model = args.model
            # f.write(f"### {scale}\n")
            # main(df, scale, model, f)
        else:
            # scales = ["StandardScaler", "MinMaxScaler"]
            scales = ["MinMaxScaler"]
            models = ["model_1", "model_2", "model_3"]

            for scale in scales:
                f.write(f"\n### {scale}\n")
                df_norm = normalization_df(df, scale)
                for model_name in models:
                    print(f"Start training {model_name}")

                    # neural_networks.kfold_train(df_norm,model_name,f)
                    neural_networks.gridsearch(df_norm, model_name, f)
