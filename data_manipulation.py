import csv
import pandas as pd
from multiprocessing import Pool
import glob
import confix


def extract_interaction_eliminate_unclassified(data):
    interaction_dict = {}
    for row in data:
        pdb_id = row[0]
        resid1 = row[2]
        resid2 = row[27]
        key = f"{pdb_id},{resid1},{resid2}"
        print(key)
        interaction = row[-1]
        if interaction == "NaN" or interaction == "":
            continue
        values = row  # remove useless values
        try:
            interaction_dict[key][-1].append(interaction)
        except KeyError:
            values[-1] = [interaction]
            interaction_dict[key] = values
    return interaction_dict


def extract_interaction_using_unclassified(data):
    processed_dict = {}
    for row in data:
        pdb_id, resid1, resid2, interaction = row[0], row[2], row[27], row[-1]
        if interaction == "NaN" or interaction == "":
            interaction = "Unclassified"
        key = f"{pdb_id},{resid1},{resid2}"
        if key in processed_dict:
            interactions = processed_dict[key][-1]
            interactions.append(interaction)
            processed_dict[key][-1] = list(set(interactions))
        else:
            processed_dict[key] = row[:-1] + [[interaction]]
    return processed_dict


def save_dict_to_tsv(dictionary, filename):
    with open(filename, "w+", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(dictionary.values())


def process_files(filename):
    with open(filename, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        data = list(reader)
        interaction_data = extract_interaction_using_unclassified(data[1:])
        save_dict_to_tsv(
            interaction_data, filename.replace("feature_ring_new", "features_ring_news")
        )
    return interaction_data


def encode_interactions(value, labels):
    value[-1] = labels[str(tuple(value[-1]))]
    return value


def generate_interaction_dataframe(function):
    interaction_data = {}
    process_files(confix.PATH_FEATURE_RING_TSV)
    with Pool(15) as p:
        for data in p.map(function, glob.glob(confix.PATH_FEATURE_RING_TSV)):
            interaction_data.update(data)

    interactions = [v[-1] for _, v in interaction_data.items()]
    unique_interactions = list(map(str, set(map(tuple, interactions))))
    interaction_labels = {k: c for c, k in enumerate(unique_interactions)}

    processed_data = {
        key: encode_interactions(value, interaction_labels)
        for key, value in interaction_data.items()
    }
    columns = [
        "pdb_id",
        "s_ch",
        "s_resi",
        "s_ins",
        "s_resn",
        "s_ss8",
        "s_rsa",
        "s_nh_relidix",
        "s_nh_energy",
        "s_o_relidx",
        "s_o_energy",
        "s_nh2_relidx",
        "s_nh2_energy",
        "s_o2_relidx",
        "s_o2_energy",
        "s_up",
        "s_down",
        "s_phi",
        "s_psi",
        "s_ss3",
        "s_a1",
        "s_a2",
        "s_a3",
        "s_a4",
        "s_a5",
        "t_ch",
        "t_resi",
        "t_ins",
        "t_resn",
        "t_ss8",
        "t_rsa",
        "t_nh_relidix",
        "t_nh_energy",
        "t_o_relidx",
        "t_o_energy",
        "t_nh2_relidx",
        "t_nh2_energy",
        "t_o2_relidx",
        "t_o2_energy",
        "t_up",
        "t_down",
        "t_phi",
        "t_psi",
        "t_ss3	",
        "t_a1",
        "t_a2",
        "t_a3",
        "t_a4",
        "t_a5",
        "Interaction",
        "nan",
    ]
    df = pd.DataFrame(list(processed_data.values()))
    df.columns = columns
    df.drop("nan", axis=1, inplace=True)
    return df
