import os
from Bio import PDB
from urllib.request import urlretrieve
import confix
import subprocess


def download_pdb(pdb_code, output_folder):
    # Check if the PDB file already exists in the output folder
    output_path = os.path.join(output_folder, f"{pdb_code}.pdb")
    if os.path.exists(output_path):
        print(f"PDB file {pdb_code}.pdb already exists in the output folder.")
        return

    # Download the PDB file from the Protein Data Bank
    pdbl = PDB.PDBList()
    pdbl.retrieve_pdb_file(pdb_code, file_format="pdb", pdir=output_folder)


def create_pdb_folder(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of .tav files in the input folder
    tav_files = [f for f in os.listdir(input_folder) if f.endswith(".tsv")]

    # Extract the PDB codes and download the PDB files
    count = 1
    for tav_file in tav_files:
        pdb_code = tav_file[:-4]  # Remove the '.tav' extension
        download_pdb(pdb_code, output_folder)
        print(f"Downloaded PDB file for {pdb_code}. [{count},{len(tav_files)}]")
        count += 1


def create_new_feature_ring():
    file_names = os.listdir(confix.PATH_OUT_PDB_FILE)

    command = "python3 calc_features.py"

    # Iterate over the file names and run the command on each file
    counter = 0
    for file_name in file_names:
        file_path = os.path.join(confix.PATH_OUT_PDB_FILE, file_name)
        cmd = f"{command} {file_path} -out_dir {confix.PATH_NEW_FEATURE_RING}"
        print(f"crateing --> [{counter}, {len(file_names)}]")
        counter += 1
        subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    input_folder = confix.PATH_FEATURES_RING
    output_folder = confix.PATH_OUT_PDB_FILE
    create_pdb_folder(input_folder, output_folder)
    create_new_feature_ring()
