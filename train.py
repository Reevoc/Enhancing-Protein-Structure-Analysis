import argparse
from datetime import datetime
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
    parser.add_argument("-k", "--kfold", help="kfold", required=False)
    parser.add_argument(
        "-o",
        "--out_name",
        help="if provided, stores model named like this",
        required=False,
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
        args.manipulation = True
    else:
        args.manipulation = (
            True if args.manipulation == "eliminate_unclassified" else False
        )

    if not args.kfold:
        args.kfold = False

    if not args.out_name:
        args.out_name = False

    if error:
        parser.print_help()
        raise SystemExit(f"\n{' '.join(error)}")

    return args


def main(index, model_name, normalization_mode, remove_unclassified, kfold, f, name):
    print(model_name, normalization_mode, remove_unclassified)
    df = prepare_data(
        index,
        remove_unclassified=remove_unclassified,
    )

    df = normalization_df(df, normalization_mode)

    if kfold == "kfold":
        neural_networks.kfold_train(df, model_name, f)
    else:
        save_model = True if name else False
        _, _, _, _, model = neural_networks.train(
            df, model_name, f, save_model=save_model, epochs=30
        )
        print(f"Saving model {name}")
        model.save(f"{name}.h5")


if __name__ == "__main__":
    args = parser()
    now = datetime.now().strftime(r"%m-%d_%H:%M")
    out = f"./results{now}.csv"

    index = build_index(
        path_pdb=conf.PATH_PDB,
        path_ring=conf.PATH_FEATURES_RING,
        path_tsv=conf.PATH_DSSP_TSV,
        generate_new=False,
        update_dssp=False,
    )

    with open(out, "w") as f:
        header = f"manipulation\tscaler\tmodel_name\taverage_accuracy\taverage_f1\taverage_precision\taverage_recall\n"
        f.write(header)

        ###### PASSING ARGUMENTS ######
        if len(sys.argv) > 1:
            print(f"out_name: {args.out_name}")
            main(
                index=index,
                model_name=args.model,
                normalization_mode=args.normalization,
                remove_unclassified=args.manipulation,
                kfold=args.kfold,
                f=f,
                name=args.out_name,
            )
        # END
        ###### DEFAULT ######
        else:
            if False:
                ######### IF GRIDSEARCH #########
                # PARAMETERS
                dropout_rate = 0.2
                epoch = 20
                manipulation = [True, False]  # remove unclassified
                models = ["model_1", "model_2", "model_3"]
                optimizer = Adam(learning_rate=0.001)
                scales = ["MinMaxScaler", "StandardScaler"]

                # RUN
                for m in manipulation:  # remove unclassified
                    for s in scales:  # scaler
                        df = prepare_data(index, remove_unclassified=m)
                        df = normalization_df(df, s)
                        for mod in models:  # models
                            f.write(f"{m}\t{s}\t{mod}\t")
                            neural_networks.train(
                                df=df,
                                model_name=mod,
                                optimizer=optimizer,
                                epochs=epoch,
                                batch_size=conf.BATCH_SIZE,
                                dropout_rate=dropout_rate,
                                f=f,
                            )
            if False:
                ######### KFOLD #########
                KFOLD = 10
                balanced = False
                batch_size = 512
                dropout_rate = 0.2
                epoch = 30
                manipulations = [False]  # remove unclassified
                models = ["model_3"]
                optimizer = Adam(learning_rate=0.001)
                scale = ["StandardScaler"]

                for m in manipulations:
                    df = prepare_data(index, remove_unclassified=m)

                    for s in scale:
                        df_norm = normalization_df(
                            df,
                            s,
                        )

                        for mod in models:
                            f.write(f"{m}\t{s}\t{mod}\t")
                            print(f"Start training {mod}\n")
                            (
                                accuracy,
                                f1,
                                precision,
                                recall,
                            ) = neural_networks.kfold_train(
                                df=df_norm,
                                model_name=mod,
                                optimizer=optimizer,
                                epochs=epoch,
                                dropout_rate=dropout_rate,
                                batch_size=batch_size,
                                kfold=KFOLD,
                                balanced=balanced,
                                f=f,
                            )
                            f.write(f"{accuracy}\t{f1}\t{precision}\t{recall}\n")

            if True:
                ######### SINGLE TESTS #########
                if True:  # new model
                    balanced = False
                    manipulation = True  # remove unclassified
                    batch_size = 512
                    dropout_rate = 0.2
                    epoch = 30
                    model_name = "model_2"
                    optimizer = Adam(learning_rate=0.001)
                    scale = "StandardScaler"

                    df = prepare_data(index, remove_unclassified=manipulation)
                    df = normalization_df(df, scale)

                    accuracy, f1, precision, recall, model = neural_networks.train(
                        df=df,
                        model_name=model_name,
                        optimizer=optimizer,
                        epochs=epoch,
                        batch_size=batch_size,
                        dropout_rate=dropout_rate,
                        f=f,
                        balanced=False,
                        save_model=True,
                    )

                    # Predict
                    model.save(f"{model_name}.h5")
                else:
                    df = prepare_data(index, remove_unclassified=True)
                    df = normalization_df(df, "StandardScaler")
                model = load_model("model_2.h5")

                labels_corr = list(set(zip(df["Interaction"], df["OrgInteraction"])))

                accuracy, precision, recall, f1 = neural_networks.test_predict(
                    df, model
                )

                print(
                    f"Accuracy: {accuracy},  Precision: {precision}, Recall: {recall},F1: {f1}"
                )
