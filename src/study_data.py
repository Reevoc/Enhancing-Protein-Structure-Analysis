# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(confusion_matrix, ax, ax_histx, ax_histy):
    # Draw the heatmap for the confusion matrix.
    im = ax.imshow(confusion_matrix, cmap="viridis")

    # Draw the marginals.
    ax_histx.bar(
        np.arange(confusion_matrix.shape[1]),
        np.sum(confusion_matrix, axis=0),
        color="green",
    )
    ax_histy.barh(
        np.arange(confusion_matrix.shape[0]),
        np.sum(confusion_matrix, axis=1),
        color="green",
    )

    # Set labels for the axes.
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    # Set labels for the histograms.
    ax_histx.set_xlabel("Sum over predicted class")
    ax_histy.set_ylabel("Sum over true class")

    # Add a colorbar.
    fig.colorbar(im, ax=ax, shrink=0.6)


# Sample confusion matrix.
cm = confusion_matrix(y_test, y_pred)

# Start with a square Figure.
fig = plt.figure(figsize=(10, 10))

# Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
# the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig.add_gridspec(
    2,
    2,
    width_ratios=(4, 1),
    height_ratios=(1, 4),
    left=0.1,
    right=0.9,
    bottom=0.1,
    top=0.9,
    wspace=0.05,
    hspace=0.05,
)

# Create the Axes.
ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

# Draw the scatter plot and marginals.
plot_confusion_matrix(confusion_matrix, ax, ax_histx, ax_histy)

plt.show()

# %%
# read all files .tsv insife data/features_ring
import glob
import csv

list_interactions = []

for file in glob.glob("data/features_ring/*.tsv"):
    with open(file, "r") as f:
        print(file)
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            interaction = row[-1] if row[-1] != "" else "Unclassified"
            list_interactions.append((row[4], row[20], interaction))
            # print(row[-1])

# %%
len(list_interactions)

# %%
import pandas as pd
import matplotlib.pyplot as plt


# Step 1: Prepare the data and create a DataFrame
data = pd.DataFrame(
    list_interactions, columns=["aminoacid1", "aminoacid2", "interaction"]
)

# Step 2: Calculate statistics
interaction_counts = data["interaction"].value_counts()

# Step 3: Visualize the distribution
plt.figure(figsize=(10, 6))
interaction_counts.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Distribution of Amino Acid Interactions")
plt.xlabel("Interaction Type")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.show()

# %%
list_interactions = []
file = "data.tsv"
with open(file, "r") as f:
    reader = csv.reader(f, delimiter=",")
    next(reader, None)
    for row in reader:
        interaction = row[-1]
        list_interactions.append((row[4], row[26], interaction))

# %%
import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame(
    list_interactions, columns=["aminoacid1", "aminoacid2", "interaction"]
)
# Step 2: Calculate statistics
interaction_counts = data["interaction"].value_counts()
print(interaction_counts)

# Step 3: Visualize the distribution
plt.figure(figsize=(10, 6))
interaction_counts.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Distribution of Amino Acid Interactions")
plt.xlabel("Interaction Type")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
import configuration as cfg
from matplotlib.colors import ListedColormap

# Assuming df is your pandas DataFrame with the "interactions" column
# For example:
# df = pd.DataFrame({'interactions': [['A', 'B', 'C'], ['B', 'C', 'D'], ['A', 'C', 'D'], ['B', 'A', 'D']]})
df = pd.DataFrame(
    list_interactions, columns=["aminoacid1", "aminoacid2", "interaction"]
)
# Create a list of unique interactions from the "interactions" column
# Create a matrix filled with zeros
matrix = []
# Iterate through each row in the DataFrame and update the matrix
for interaction1 in list(cfg.INTERACTION_TYPES):
    lista = []
    for interaction2 in list(cfg.INTERACTION_TYPES):
        if interaction1 == interaction2:
            lista.append(
                sum([1 if interaction1 in x else 0 for x in df["interaction"]])
            )
        else:
            lista.append(
                sum(
                    [
                        1 if interaction1 in x and interaction2 in x else 0
                        for x in df["interaction"]
                    ]
                )
            )
    matrix.append(lista)

matrix = np.array(matrix)
print(matrix)
labels = list(cfg.INTERACTION_TYPES)
plt.imshow(matrix, cmap="Blues", interpolation="nearest", aspect="auto")

# Add text annotations with the counts in each square
for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")

plt.colorbar(label="Count")
plt.xticks(np.arange(len(labels)), labels, rotation=90)
plt.yticks(np.arange(len(labels)), labels)
plt.title("Matrix of interactions")
plt.show()


# %%
