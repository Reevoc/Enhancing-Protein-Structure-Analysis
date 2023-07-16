import csv
import pandas as pd
from multiprocessing import Pool
import glob
import confix
import os


def extract_interaction_eliminate_unclassified(path):
    data = pd.read_csv(path, sep="\t")
    interaction_dict = {}
    for _, row in data.iterrows():
        pdb_id = row[0]
        resid1 = row[2]
        resid2 = row[18]
        key = f"{pdb_id},{resid1},{resid2}"
        interaction = row[-1]
        if key in interaction_dict:
            interaction_dict[key][-1].append(interaction)
            interaction_dict[key][-1] = list(set(interaction_dict[key][-1]))
        else:
            interaction_dict[key] = row[:-1].tolist() + [[interaction]]
    return interaction_dict


def extract_interaction_eliminate_unclassified_second(path):
    data = pd.read_csv(path, sep="\t")
    interaction_dict = {}
    for _, row in data.iterrows():
        pdb_id = row[0]
        resid1 = row[2]
        resid2 = row[26]
        key = f"{pdb_id},{resid1},{resid2}"
        interaction_dict[key] = row.tolist()
    return interaction_dict


def extract_interaction_using_unclassified(file):
    data = pd.read_csv(file, sep="\t")
    processed_dict = {}
    for _, row in data:
        pdb_id, resid1, resid2, interaction = row[0], row[2], row[18], row[-1]
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


# Cerate he dict without appending the interactios column to the dict
def extract_interaction_using_unclassified_second(path):
    data = pd.read_csv(path, sep="\t")
    interaction_dict = {}
    for _, row in data.iterrows():
        pdb_id = row[0]
        resid1 = row[2]
        resid2 = row[26]
        key = f"{pdb_id},{resid1},{resid2}"
        interaction_dict[key] = row
    return interaction_dict


def save_dict_to_tsv(dictionary, filename):
    with open(filename, "w+", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(dictionary.values())


def process_files(filename, function):
    with open(filename, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        data = list(reader)
        interaction_data = function(data[1:])
        save_dict_to_tsv(interaction_data, filename)
    return interaction_data


def encode_interactions(value, labels):
    value[-1] = labels[str(tuple(value[-1]))]
    return value


def generate_interaction_dataframe(function):
    set1 = set(
        list(map(os.path.basename, list(glob.glob(confix.PATTERN_FEATURE_RING_TSV))))
    )
    set2 = set(
        list(
            map(os.path.basename, list(glob.glob(confix.PATTERN_FEATURE_RING_TSV_NEW)))
        )
    )
    set3 = set1.intersection(set2)

    path = list(map(lambda x: os.path.join(confix.PATH_FEATURES_RING_NEW, x), set3))

    for files in path:
        process_files(files, extract_interaction_eliminate_unclassified_second)
    print("dicted files features ring new...")

    interaction_data = {}
    count = 0
    for file in path:
        try:
            interaction_data.update(process_files(files, function))
        except Exception as e:
            count += 1
            print(f"dict problem:{file} error file number [{count}, {len(file)}]")

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
    ]
    df = pd.DataFrame(list(processed_data.values()))
    df.columns = columns
    return df


if __name__ == "__main__":
    set1 = set(
        list(map(os.path.basename, list(glob.glob(confix.PATTERN_FEATURE_RING_TSV))))
    )
    set2 = set(
        list(
            map(os.path.basename, list(glob.glob(confix.PATTERN_FEATURE_RING_TSV_NEW)))
        )
    )
    set3 = set1.intersection(set2)

    path_new = list(map(lambda x: os.path.join(confix.PATH_FEATURES_RING_NEW, x), set3))
    path_normal = list(map(lambda x: os.path.join(confix.PATH_FEATURES_RING, x), set3))

    full_dict_eliminate_unclassified_second = {}
    for files in path_new:
        dict = extract_interaction_eliminate_unclassified_second(files)
        full_dict_eliminate_unclassified_second.update(dict)

    print(f"keys: {len(full_dict_eliminate_unclassified_second.keys())}")
    print(full_dict_eliminate_unclassified_second["1aba,84,87"])

    full_dict_eliminate_unclassified = {}
    for files in path_normal:
        dict = extract_interaction_eliminate_unclassified(files)
        full_dict_eliminate_unclassified.update(dict)

    print(f"keys: {len(full_dict_eliminate_unclassified.keys())}")
    print(full_dict_eliminate_unclassified["1aba,39,42"])

    for key in full_dict_eliminate_unclassified_second:
        if key in full_dict_eliminate_unclassified:
            full_dict_eliminate_unclassified_second[key].append(
                full_dict_eliminate_unclassified[key][-1]
            )

    print(full_dict_eliminate_unclassified_second["1aba,39,42"])

    df = pd.DataFrame(full_dict_eliminate_unclassified_second.values())
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
    ]
    df.dropna(inplace=True)
    df.columns = columns
