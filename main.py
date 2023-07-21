import data_manipulation
import normalization
from keras.utils import to_categorical

import neural_networks
from myparser import parser


def main(manipulation, scale, model, path_output="./results.csv"):
    df = data_manipulation.generate_data(manipulation == "eliminate_unclassified")
    # df = pd.read_csv("./merged.tsv", sep="\t")

    df = normalization.all_normalization(df, scale)

    loss, accuracy = neural_networks.model(df, model)
    if path_output:
        with open("./results.csv", "a") as f:
            f.write(
                f"manipulation: {manipulation}, model: {model}, normalization: {scale}, loss: {loss}, accuracy: {accuracy}\n"
            )


if __name__ == "__main__":
    args = parser()
    if args:
        manipulation = args.manipulation
        scale = args.normalization
        model = args.model
        out = args.out
        main(manipulation, scale, model, out)
    else:
        # get current time
        import datetime

        now = datetime.datetime.now().strftime("%H:%M")
        out = f"./results{now}.csv"
        models = ["model_1", "model_2", "model_3"]
        scales = ["MinMaxScaler", "StandardScaler", "no_normalization"]
        manipulations = ["eliminate_unclassified", "unclassified"]
        for model in models:
            for scale in scales:
                for manipulation in manipulations:
                    main(manipulation, scale, model, out)
