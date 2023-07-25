# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

with open("output.csv", "r") as f:
    df = pd.read_csv(f, delimiter="\t")
    y_test = df["Y_test"]
    y_pred = df["Y_pred"]
    y_test = y_test.to_numpy()
    y_pred = y_pred.to_numpy()

with open("label.csv", "r") as f:
    next(f)
    lines = set([l for l in f])
    set_labels = [list(map(str.strip, l.split("\t"))) for l in lines]
    set_labels = map(lambda x: (int(x[0]), x[1]), set_labels)
    set_labels = sorted(set_labels, key=lambda x: x[0])

print(set_labels)


def plot_confusion_matrix(
    cm, ax, ax_histx, ax_histy
):  # Use 'cm' instead of 'confusion_matrix'
    # Draw the heatmap for the confusion matrix.

    im = ax.imshow(cm, cmap="viridis")

    for i in range(len(set_labels)):
        for j in range(len(set_labels)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="white")

    # Draw the marginals.
    ax_histx.bar(
        np.arange(cm.shape[1]),
        np.sum(cm, axis=0),
        color="green",
    )
    # change hsitogram name x name
    ax_histy.barh(
        np.arange(cm.shape[0]),
        np.sum(cm, axis=1),
        color="green",
    )
    set_labes_indices = [x[0] for x in set_labels]
    set_labes_name = [x[1] for x in set_labels]
    print(set_labes_name)
    ax.set_xticks(set_labes_indices)
    # Set labels for the axes.
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    # Set labels for the histograms.
    ax_histx.set_xlabel("Sum over predicted class")
    ax_histy.set_ylabel("Sum over true class")
    ax_histx.set_xticks(set_labes_indices)
    ax_histy.set_xticks(set_labes_indices)
    ax_histx.set_yticks(set_labes_indices)
    ax_histy.set_yticks(set_labes_indices)

    # Add a colorbar.
    fig.colorbar(im, ax=ax, shrink=0.6)


# Calculate the confusion matrix.
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
plot_confusion_matrix(
    cm, ax, ax_histx, ax_histy
)  # Pass 'cm' instead of 'confusion_matrix'

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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pandas as pd

# Read the data from output.csv
with open("output_grande.csv", "r") as f:
    df = pd.read_csv(f, delimiter="\t")
    y_test = df["Y_test"]
    y_pred = df["Y_pred"]
    y_test = y_test.to_numpy()
    y_pred = y_pred.to_numpy()

print(len(set(y_test)))
print(len(set(y_pred)))
print(set(y_test))
print(set(y_pred))


with open("labels_grande.csv", "r") as f:
    next(f)
    lines = set([l for l in f])
    set_labels = [list(map(str.strip, l.split("\t"))) for l in lines]
    set_labels = map(lambda x: (int(x[0]), x[1]), set_labels)
    set_labels = sorted(set_labels, key=lambda x: x[0])

print(set_labels)

Blues = cm.get_cmap("Blues", 1000)
newcolors = Blues(np.linspace(0, 1, 1000))
newcmp = ListedColormap(newcolors[:700])


def plot_confusion_matrix(cm, ax, ax_histx, ax_histy):
    # Draw the heatmap for the confusion matrix.
    im = ax.imshow(cm, cmap=newcmp, interpolation="nearest", aspect="auto")

    for i in range(len(set_labels)):
        print("i loop")
        print(cm[i, :])
        for j in range(len(set_labels)):
            print("j loop")
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="Black")

    # Draw the marginals.
    ax_histx.bar(
        np.arange(cm.shape[1]),
        np.sum(cm, axis=0),
        color="#99ccff",
    )
    ax_histy.barh(
        np.arange(cm.shape[0]),
        np.sum(cm, axis=1),
        color="#99ccff",
    )
    set_labels_indices = [x[0] for x in set_labels]
    set_labels_names = [str(x[0]) for x in set_labels]
    
     # Use names for the tick labels
    # Use names for the tick labels
    ax_histx.set_xticks(set_labels_indices)
    # Use names for the tick labels
    ax_histy.set_yticks(set_labels_indices)  # Use names for the tick labels
    # Set labels for the axes.
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    # Set labels for the histograms.
    ax_histx.set_xlabel("Sum over predicted class")
    ax_histy.set_ylabel("Sum over true class")
    ax.set_xticks(set_labels_indices)  # Use names for the tick labels
    ax.set_yticks(set_labels_indices)
    ax.set_xtickslabels(set_labels_names) 
    ax.set_ytickslabels(set_labels_names)

    # Add a colorbar.
    fig.colorbar(im, ax=ax, shrink=0.6)


# Calculate the confusion matrix.
cm = confusion_matrix(y_test, y_pred)

# Start with a square Figure.
fig = plt.figure(figsize=(14, 14))

# Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
# the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig.add_gridspec(
    2,
    2,
    width_ratios=(5, 1),
    height_ratios=(1, 5),
    left=0.1,
    right=1,
    bottom=0.1,
    top=1,
    wspace=0.10,
    hspace=0.10,
)

# Create the Axes.
ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

# Draw the scatter plot and marginals.
plot_confusion_matrix(cm, ax, ax_histx, ax_histy)

#####
# After your existing code...
# Create a new figure for the table.


# Create data for the table where the first column is the indices and the second column is the labels.

# Hide axes.
ax.axis("off")
####

plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pandas as pd
import matplotlib.gridspec as gridspec

# ... Your data loading code here ...
# Read the data from output.csv
with open("output.csv", "r") as f:
    df = pd.read_csv(f, delimiter="\t")
    y_test = df["Y_test"]
    y_pred = df["Y_pred"]
    y_test = y_test.to_numpy()
    y_pred = y_pred.to_numpy()

with open("label.csv", "r") as f:
    next(f)
    lines = set([l for l in f])
    set_labels = [list(map(str.strip, l.split("\t"))) for l in lines]
    set_labels = map(lambda x: (int(x[0]), x[1]), set_labels)
    set_labels = sorted(set_labels, key=lambda x: x[0])

print(set_labels)


Blues = cm.get_cmap("Blues", 1000)
newcolors = Blues(np.linspace(0, 1, 1000))
newcmp = ListedColormap(newcolors[:700])


# ... Your plot_confusion_matrix function definition here ...
def plot_confusion_matrix(cm, ax, ax_histx, ax_histy):
    # Draw the heatmap for the confusion matrix.
    im = ax.imshow(cm, cmap=newcmp, interpolation="nearest", aspect="auto")

    for i in range(len(set_labels)):
        for j in range(len(set_labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="Black")

    # Draw the marginals.
    ax_histx.bar(
        np.arange(cm.shape[1]),
        np.sum(cm, axis=0),
        color="#99ccff",
    )
    ax_histy.barh(
        np.arange(cm.shape[0]),
        np.sum(cm, axis=1),
        color="#99ccff",
    )
    set_labels_indices = [x[0] for x in set_labels]
    set_labels_names = [x[1] for x in set_labels]
    ax.set_xticks(set_labels_indices)  # Use names for the tick labels
    ax.set_yticks(set_labels_indices)
    # Use names for the tick labels
    ax_histx.set_xticks(set_labels_indices)
    ax.set_xticklabels(set_labels_indices)
    ax.set_yticklabels(set_labels_indices
    # Use names for the tick labels
    ax_histy.set_yticks(set_labels_indices)  # Use names for the tick labels

    # Set labels for the axes.
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    # Set labels for the histograms.
    ax_histx.set_xlabel("Sum over predicted class")
    ax_histy.set_ylabel("Sum over true class")

    # Add a colorbar.
    fig.colorbar(im, ax=ax, shrink=0.6)


# Calculate the confusion matrix.
cm = confusion_matrix(y_test, y_pred)

# Create a new figure.
fig = plt.figure(figsize=(14, 14))

# Set the grid layout for the figure.
gs = gridspec.GridSpec(3, 2, height_ratios=[4, 1, 0.2], hspace=0.3)

# Add your plot to the first subplot.
ax = plt.subplot(gs[0, :])
ax_histx = plt.subplot(gs[1, 0], sharex=ax)
ax_histy = plt.subplot(gs[1, 1], sharey=ax)

# Draw the scatter plot and marginals.
plot_confusion_matrix(cm, ax, ax_histx, ax_histy)

# Add a table to the third subplot.
ax2 = plt.subplot(gs[2, :])

# Create data for the table where the first column is the indices and the second column is the labels.
table_data = [[idx, label] for idx, label in set_labels]

# Add a table at the center of the axes.
the_table = ax2.table(cellText=table_data, colLabels=["Index", "Label"], loc="center")

# Hide axes.
ax2.axis("off")

# Show the plot and table.
plt.tight_layout()
plt.show()
