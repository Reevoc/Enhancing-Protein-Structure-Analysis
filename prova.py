import os 
import sys
import pandas as pd
import numpy as np
import confix
from sklearn import preprocessing
import time
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
#import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
#import tensorflow as tf
from IPython.display import display, HTML

# Combine all PDBs into a single dataframe
dfs = []
for filename in os.listdir(confix.PATH_FEATURES_RING):
    dfs.append(pd.read_csv(confix.PATH_FEATURES_RING + filename, sep='\t'))
df = pd.concat(dfs)