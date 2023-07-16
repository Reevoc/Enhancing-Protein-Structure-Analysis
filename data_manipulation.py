import csv
import pandas as pd
from multiprocessing import Pool
import glob
import confix
import os


def extract_interaction_FR(path):
    data = pd.read_csv(path, sep="\t")
    interaction_dict = {}
    for _, row in data.iterrows():
        pdb_id = row[0]
        resid1 = row[2]
        resid2 = row[18]
        key = f"{pdb_id},{resid1},{resid2}"
        interaction = row[-1]
        if interaction == "NaN" or interaction == "":
            interaction = "Unclassified"
        if key in interaction_dict:
            interaction_dict[key][-1].append(interaction)
            interaction_dict[key][-1] = list(set(interaction_dict[key][-1]))
        else:
            interaction_dict[key] = row[:-1].tolist() + [[interaction]]
    return interaction_dict


def extract_interaction_FRN(path):
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


def get_common_set(pattern1, pattern2):
    set1 = set(list(map(os.path.basename, list(glob.glob(pattern1)))))
    set2 = set(list(map(os.path.basename, list(glob.glob(pattern2)))))
    return set1.intersection(set2)


def get_file_paths(directory, filenames):
    return list(map(lambda x: os.path.join(directory, x), filenames))


def merge_dictionaries(dict1, dict2):
    merged_dict = {}
    for key in dict1:
        if key in dict2:
            merged_dict[key] = dict1[key] + [dict2[key][-1]]
    return merged_dict


def extract_data_from_files(paths, extraction_function):
    result_dict = {}
    for file_path in paths:
        file_dict = extraction_function(file_path)
        result_dict.update(file_dict)
    return result_dict


def generate_data():
    set3 = get_common_set(
        confix.PATTERN_FEATURE_RING_TSV, confix.PATTERN_FEATURE_RING_TSV_NEW
    )
    path_new = get_file_paths(confix.PATH_FEATURES_RING_NEW, set3)
    path_normal = get_file_paths(confix.PATH_FEATURES_RING, set3)

    dict_FRN = extract_data_from_files(path_new, extract_interaction_FRN)
    print(f"keys: {len(dict_FRN.keys())}")
    print(dict_FRN["1aba,84,87"])

    dict_FR = extract_data_from_files(path_normal, extract_interaction_FR)
    print(f"keys: {len(dict_FR.keys())}")
    print(dict_FR["1aba,39,42"])

    merged_dict = merge_dictionaries(dict_FRN, dict_FR)
    print(merged_dict["1aba,39,42"])

    df = pd.DataFrame(merged_dict.values())
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
    df.to_csv("./merged.tsv", sep="\t", index=False)
    return df
