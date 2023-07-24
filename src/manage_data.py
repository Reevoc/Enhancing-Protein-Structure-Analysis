import csv
import glob
import os
from multiprocessing import Pool

import pandas as pd
from Bio.PDB import PDBList

import configuration as conf
from src.features import generate_feature_file


def _download_cif(x):
    pdbl = PDBList(obsolete_pdb=os.path.join(conf.PATH_PDB, "obsolete"))
    pdbl.retrieve_pdb_file(pdb_code=x, file_format="mmCif", pdir=conf.PATH_PDB)


def build_index(path_pdb, path_tsv, path_ring, generate_new=False, update_dssp=False):
    print("Building Index")
    # Fetch list of ids from given files with interaction column
    ring_files = list(glob.glob(path_ring + "/*.tsv"))
    id_from_ring = set(map(lambda x: os.path.basename(x).split(".")[0], ring_files))

    # Fetch list of downloaded pdbs
    pdb_accepted_files = ["*.tsv", "*.ent", "*.cif"]
    pdb_files = []
    for ext in pdb_accepted_files:
        pdb_files.extend(list(glob.glob(path_pdb + "/" + ext)))
    pdb_files = set(pdb_files)
    id_from_pdb = set(map(lambda x: os.path.basename(x).split(".")[0], pdb_files))

    # Fetch list of locally generate features files
    tsv_files = list(glob.glob(path_tsv + "/*.tsv"))
    id_from_tsv = set(map(lambda x: os.path.basename(x).split(".")[0], tsv_files))

    # --------------------------------
    list_of_ids = id_from_ring
    missing_pdb = id_from_ring.union(id_from_tsv) - id_from_pdb
    missing_tsv = id_from_ring if update_dssp else id_from_ring - id_from_tsv
    missing_tsv = missing_tsv - missing_pdb

    with open(os.path.join(conf.PATH_PDB, "missing_pdb.txt"), "w+") as f:
        f.writelines(",".join(missing_pdb))

    if generate_new:
        errors = []
        with Pool(conf.MTHREAD) as p, open("dssp_error.txt", "w+") as f:
            print("Generating features files")
            print(missing_tsv)
            for pdb_error in p.map(generate_feature_file, missing_tsv):
                if pdb_error:
                    errors.append(pdb_error)
                    f.write(pdb_error + "\n")
        with open(os.path.join(conf.PATH_PDB, "dssp_errors.txt"), "w+") as f:
            f.writelines(",".join(errors))
        missing_tsv = missing_tsv.intersection(set(errors))

    list_of_ids = (list_of_ids - missing_pdb) - missing_tsv

    index = {
        id: {
            "pdb": os.path.join(path_pdb, f"{id}.cif"),
            "tsv": os.path.join(path_tsv, f"{id}.tsv"),
            "ring": os.path.join(path_ring, f"{id}.tsv"),
        }
        for id in list_of_ids
    }

    with open(os.path.join(conf.PATH_PDB, "pdbis.txt"), "w+") as f:
        f.writelines(",".join(list_of_ids))

    print("End of building index")
    return index


def check_resi():
    print("\tChecking resi")
    # Use glob to find all TSV files
    tsv_files = glob.glob(os.path.join(conf.PATH_DSSP_TSV, "*.tsv"))
    list_pdbs = []
    # Iterate over the TSV files
    for file_path in tsv_files:
        with open(file_path, "r") as file:
            tsv_reader = csv.reader(file, delimiter="\t")
            next(tsv_reader, None)

            # Read each row in the TSV file
            for row in tsv_reader:
                t_resi = row[
                    conf.COLUMN_DSSP.index("t_resi")
                ]  # Index 21 corresponds to t_resi
                s_resi = row[
                    conf.COLUMN_DSSP.index("s_resi")
                ]  # Index 3 corresponds to s_resi

                # Check if t_resi and s_resi are integers
                if t_resi.isdigit() and s_resi.isdigit():
                    # print(f"In file {file_path}: Both t_resi and s_resi are integers.")
                    pass
                else:
                    list_pdbs.append(os.path.basename(file_path)[:-3])
                    print(len(conf.COLUMN_DSSP), len(row))
                    print(list(zip(conf.COLUMN_DSSP, row)))
                    print(
                        f"In file {file_path}: Either t_resi or s_resi is not an integer."
                    )

    list_pdbs = set(list_pdbs)
    print(list_pdbs)
    print(len(list_pdbs))


def _import_tsv(path, delete_unclassified=False):
    temp = {}
    is_ring = "ring" in path
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if is_ring:
                pdb_id, resid1, resid2, interaction = row[0], row[2], row[18], row[-1]
                if interaction == "NaN" or interaction == "":
                    if delete_unclassified:
                        continue
                    else:
                        interaction = "Unclassified"
                key = f"{pdb_id},{resid1},{resid2}"
                if key in temp:
                    interactions = temp[key][-1]
                    interactions.append(interaction)
                    temp[key][-1] = list(set(interactions))
                else:
                    temp[key] = row[:-1] + [[interaction]]
            else:
                pdb_id, resid1, resid2 = row[0], row[2], row[26]
                key = f"{pdb_id},{resid1},{resid2}"
                temp[key] = row

    temp = {
        k: v
        for k, v in temp.items()
        if not all(map(lambda x: not isinstance(x, str), v))
    }

    return temp


def _import_data(index: dict, delete_unclassified=False) -> dict:
    ring_dict = {}
    tsv_dict = {}
    with Pool(conf.MTHREAD) as p:
        print("\timporting ring features")
        for t in p.map(_import_tsv, list(map(lambda x: x["ring"], index.values()))):
            ring_dict.update(t)
        print("\timporting tsv features")
        for t in p.map(_import_tsv, list(map(lambda x: x["tsv"], index.values()))):
            tsv_dict.update(t)

    print(
        f"length of tsv is {len(list(tsv_dict.keys()))} and ring {len(list(ring_dict.keys()))}"
    )

    # Append interactions
    for k in ring_dict.keys():
        try:
            interaction = ring_dict[k][-1]
            if interaction:  # TODO removes "None" interactions in some files
                tsv_dict[k].append(ring_dict[k][-1])
        except:
            pass

    # remove from tsv_dict keys not present in ring_dict
    tsv_dict = {k: v for k, v in tsv_dict.items() if k in ring_dict}
    return tsv_dict


def print_id_removed_column(removed):
    id_counts = {}
    for item in removed:
        id = item.split(",")[0]
        id_counts[id] = id_counts.get(id, 0) + 1

    ids_set = set(id_counts.keys())

    for id in ids_set:
        count = id_counts[id]
        print(f"ID: {id}, Count: {count}")


def prepare_data(index, remove_unclassified=True) -> pd.DataFrame:
    print("Processing data")
    newdic = _import_data(index)

    df = pd.DataFrame(
        newdic.values(),
        columns=conf.COLUMNS_BIG,
    )
    print(df.head())

    if remove_unclassified:
        print("\tremove unclassified")
        df = df[~df["Interaction"].apply(lambda x: "Unclassified" in x)]

    return df
