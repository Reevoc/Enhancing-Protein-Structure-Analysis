import pandas as pd
import os
import confix
import subprocess

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

counter = 0
file_names = os.listdir(confix.PATH_NEW_FEATURE_RING)

for file_name in file_names:
    file1_path = os.path.join(confix.PATH_FEATURES_RING, file_name)
    file2_path = os.path.join(confix.PATH_NEW_FEATURE_RING, file_name)

    try:
        df1 = pd.read_csv(file1_path, sep="\t")
        df2 = pd.read_csv(file2_path, sep="\t")

        merged_file_path = os.path.join(confix.PATH_NEW_FEATURE_RING, file_name)
        print(f"Append Interactions --> [{counter}, {len(file_names)}]")
        counter += 1
        df2.to_csv(merged_file_path, sep="\t", index=False)
    except Exception as e:
        print(f"Error occurred while merging file '{file_name}': {str(e)}")
        continue
