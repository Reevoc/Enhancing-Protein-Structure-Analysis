import itertools
import subprocess

# Define the possible options
models = ["model_1", "model_2", "model_3"]
normalizations = ["MinMaxScaler", "StandardScaler", "no_normalization"]
data_options = ["eliminate_unclassified", "unclassified"]

# Generate all combinations of options
combinations = list(itertools.product(models, normalizations, data_options))

# Iterate over the combinations and run the terminal command
for combination in combinations:
    model, normalization, data = combination
    command = f"python3 main.py -m {model} -n {normalization} -d {data}"
    subprocess.run(command, shell=True)
