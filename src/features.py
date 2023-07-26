import os

import pandas as pd
from Bio.PDB import DSSP, HSExposureCB, NeighborSearch, PPBuilder, is_aa
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.SeqUtils import seq1

import configuration as conf

# Ramachandran regions
regions_matrix = []
with open(conf.RAMA_FILE) as f:
    for line in f:
        if line:
            regions_matrix.append([int(ele) for ele in line.strip().split()])

# Atchely scales
atchley_scale = {}
with open(conf.ATCHLEY_FILE) as f:
    next(f)
    for line in f:
        line = line.strip().split("\t")
        atchley_scale[line[0]] = line[1:]


def filter_condition(value):
    s_resi = value[conf.COLUMNS_BIG.index("s_resi")]
    t_resi = value[conf.COLUMNS_BIG.index("t_resi")]
    if s_resi == -1 or isinstance(s_resi, str):
        return False
    if t_resi == -1 or isinstance(t_resi, str):
        return False

    column_angles = ["s_phi", "s_psi", "t_psi", "t_phi"]
    indices = [conf.COLUMNS_BIG.index(x) for x in column_angles]
    for index in indices:
        if isinstance(value[index], str):
            return False

    return True


def generate_feature_file(pdb_id, path=conf.PATH_PDB, write=True):
    ISERROR = False
    # Load the PDB structure
    print("generating features for", pdb_id)
    pdb_file = os.path.join(path, f"{pdb_id}.cif")

    assert "cif" in pdb_file
    structure = MMCIFParser(QUIET=True).get_structure(pdb_id, pdb_file)

    # Get valid residues
    residues = [
        residue
        for residue in structure[0].get_residues()
        if is_aa(residue) and residue.id[0] == " "
    ]

    if not residues:
        print(pdb_id, "no valid residues")
        raise ValueError("no valid residues")

    # Calculate DSSP
    try:
        dssp = dict(DSSP(structure[0], pdb_file, conf.PATH_DSSP))
    except Exception:
        print("{} DSSP error".format(pdb_id))
        dssp = {}
        ISERROR = True

    # Calculate Half Sphere Exposure
    try:
        hse = dict(HSExposureCB(structure[0]))
    except Exception:
        print("{} HSE error".format(pdb_id))
        hse = {}
        ISERROR = True

    # Calculate Ramachandran values
    rama_dict = {}
    ppb = PPBuilder()
    for chain in structure[0]:
        for pp in ppb.build_peptides(chain):
            phi_psi = pp.get_phi_psi_list()
            for i, residue in enumerate(pp):
                phi, psi = phi_psi[i]
                ss_class = None
                if phi is not None and psi is not None:
                    for x, y, width, height, ss_c, color in conf.RAMA_SS_RANGES:
                        if x <= phi < x + width and y <= psi < y + height:
                            ss_class = ss_c
                            break
                rama_dict[(chain.id, residue.id)] = [phi, psi, ss_class]

    # Generate contacts and add features
    data = []
    ns = NeighborSearch([atom for residue in residues for atom in residue])
    for residue_1, residue_2 in ns.search_all(conf.DISTANCE_THRESHOLD, level="R"):
        index_1 = residues.index(residue_1)
        index_2 = residues.index(residue_2)

        if abs(index_1 - index_2) >= conf.SEQUENCE_SEPARATION:
            aa_1 = seq1(residue_1.get_resname())
            aa_2 = seq1(residue_2.get_resname())
            chain_1 = residue_1.get_parent().id
            chain_2 = residue_2.get_parent().id

            t = (
                pdb_id,  # pdb_id
                chain_1,  # s_ch
                *residue_1.id[1:],  # s_resi
                # None,
                aa_1,  # s_ins
                *dssp.get((chain_1, residue_1.id), [None, None])[2:4],
                # s_resn, s_ss8
                *dssp.get(
                    (chain_1, residue_1.id),
                    [None, None, None, None, None, None, None, None],
                )[
                    6:14
                ],  #
                *hse.get((chain_1, residue_1.id), [None, None])[:2],
                # s_up, s_down (hals sphere exposure)
                *rama_dict.get((chain_1, residue_1.id), [None, None, None]),
                # s_phi, s_psi, s_ss3
                *atchley_scale[aa_1],  # s_a1, s_a2, s_a3, s_a4, s_a5
                chain_2,  # chain id
                *residue_2.id[1:],  # resiude id
                aa_2,  # a_aa2
                *dssp.get((chain_2, residue_2.id), [None, None])[2:4],
                #
                *dssp.get(
                    (chain_1, residue_1.id),
                    [None, None, None, None, None, None, None, None],
                )[
                    6:14
                ],  #
                *hse.get((chain_2, residue_2.id), [None, None])[:2],
                # s_up, s_down
                *rama_dict.get((chain_2, residue_2.id), [None, None, None]),
                *atchley_scale[aa_2],
            )
            if None not in t:
                data.append(t)

    # Check for errors
    filtered_data = [value for value in data if filter_condition(value)]

    if not filtered_data:
        print("{} no contacts error (skipping prediction)".format(pdb_id))
        ISERROR = True

    if ISERROR:
        return pdb_id

    # Create a DataFrame and save to file
    df = pd.DataFrame(filtered_data, columns=conf.COLUMN_DSSP).round(3)

    output_file = os.path.join(conf.PATH_DSSP_TSV, f"{pdb_id}.tsv")
    print(output_file)
    if write:
        with open(output_file, "w+") as f:
            df.to_csv(f, sep="\t", index=False)

    return df
