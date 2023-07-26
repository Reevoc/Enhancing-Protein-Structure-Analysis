import argparse
import os
import re
from datetime import datetime
from glob import glob

import pandas as pd
from keras.models import load_model
from keras.optimizers import Adam

import configuration as conf
import src.neural_networks as neural_networks
from src.features import generate_feature_file
from src.manage_data import build_index, download_cif
from src.normalization import normalization_df


def parser():
    parser = argparse.ArgumentParser(
        description="prediction of the interactions for RING software"
    )
    parser.add_argument("-p", "--pdbid", help="pdb_id protein", required=False)
    parser.add_argument(
        "-f", "--folder", help="folder containning files", required=False
    )
    parser.add_argument("-m", "--model", help="model name", required=True)
    parser.add_argument("-o", "--output", help="output file", required=False)

    args = parser.parse_args()
    pdb_id_pattern = re.compile(r"^[a-zA-Z0-9]{4}$")

    # Check if the provided pdb_id matches the pattern

    error = []
    if args.pdbid is None and args.folder is None:
        parser.print_help()
        raise SystemExit("pdb_id or folder is required.")

    if args.pdbid:
        if not bool(pdb_id_pattern.match(args.pdbid)):
            raise SystemExit("pdb_id is not valid.")

    if not os.path.isfile(args.model):
        raise SystemExit("model is not found.")

    if not args.folder:
        args.folder = None

    if not args.output:
        args.output = "output.out"

    return args


def transform_strings(s):
    # Remove curly braces and single quotes from the string
    s = s.strip("'{}'")

    # If there are multiple items separated by commas, split them and join with a colon
    if "," in s:
        items = s.split("', '")
        s = ":".join(items)

    return s


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
    df = None
    if args.folder:
        for fcif in glob(f"{args.folder}/*.cif"):
            tdf = generate_feature_file(
                fcif.split("/")[-1][:-4], args.folder, write=False
            )

            if df is None:
                df = tdf
            df = pd.concat([df, tdf], ignore_index=True)
    elif args.pdbid:
        print(f"{args.pdbid} not in index")
        path_pdb = conf.PATH_PDB
        path_tsv = conf.PATH_DSSP_TSV
        download_cif(args.pdbid, path_pdb)
        df = generate_feature_file(args.pdbid, write=False)
        index[args.pdbid] = {
            "pdb": os.path.join(path_pdb, f"{args.pdbid}.cif"),
            "tsv": os.path.join(path_tsv, f"{args.pdbid}.tsv"),
            "ring": None,
        }

    print(df.head())
    with open(out, "w") as f:
        header = "s_aa\tt_aa\tInteraction type"
        f.write(header)

        df_norm = normalization_df(df, "StandardScaler", infere=True)

        model = load_model(args.model)
        y_pred = neural_networks.infere(df_norm, model)

        print(y_pred)

        # write in output.out the y pred line by line transformed into labels string taken from labelorg.csv with \t delimiter
        with open("labelorg.csv", "r") as ff, open(args.output, "w") as out:
            next(ff)
            labels = {}
            for line in ff:
                line = line.split("\t")
                labels[int(line[0])] = line[1].strip()
            print(labels)
            for (_, r), y in zip(df.iterrows(), y_pred):
                out.write(
                    f"{r['pdb_id']}\t{r['s_resi']}\t{r['t_resi']}\t{transform_strings(labels[y])}\n"
                )
