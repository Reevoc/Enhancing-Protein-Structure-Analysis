import os
from Bio import PDB
from urllib.request import urlretrieve
import confix


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


if __name__ == "__main__":
    input_folder = confix.PATH_FEATURES_RING
    output_folder = confix.PATH_OUT_PDB_FILE
    create_pdb_folder(input_folder, output_folder)
