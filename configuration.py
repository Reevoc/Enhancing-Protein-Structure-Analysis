# Path to data Files, path to external tools and hyperparameters
import os
from multiprocessing import cpu_count

DEBUG = False
MTHREAD = cpu_count() * 2
KFOLDS = 2
BATCH_SIZE = 512
EPOCHS = 15

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_DIR)
PATH_DSSP = os.path.join(ROOT_DIR, "src/mkdssp")  # Path to DSSP executable
PATH_FEATURES_RING = os.path.join(ROOT_DIR, "data/features_ring")
PATH_PDB = os.path.join(ROOT_DIR, "data/pdb")
# PATH_NEW_FEATURE_RING = os.path.join(ROOT_DIR, "data/feature_ring_new/")
PATH_DSSP_TSV = os.path.join(ROOT_DIR, "data/tsv")

RAMA_FILE = "data/ramachandran.dat"
ATCHLEY_FILE = "data/atchley.tsv"
DISTANCE_THRESHOLD = 3.5
SEQUENCE_SEPARATION = 3
RAMA_SS_RANGES = [
    [-180, -180, 80, 60, "E", "blue"],
    [-180, 50, 80, 130, "E", "blue"],
    [-100, -180, 100, 60, "P", "green"],
    [-100, 50, 100, 130, "P", "green"],
    [-180, -120, 180, 170, "H", "red"],
    [0, -180, 180, 360, "L", "yellow"],
]

INTERACTION_TYPES = [
    "HBOND",
    "VDW",
    "PIPSTACK",
    "IONIC",
    "SSBOND",
    "PICATION",
    "Unclassified",
]


COLUMS_SMALL = [
    "pdb_id",
    "s_ch",
    "s_resi",
    "s_ins",
    "s_resn",
    "s_ss8",
    "s_rsa",
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
    "t_up",
    "t_down",
    "t_phi",
    "t_psi",
    "t_ss3",
    "t_a1",
    "t_a2",
    "t_a3",
    "t_a4",
    "t_a5",
    "Interaction",
]

COLUMNS_BIG = [
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
    "t_ss3",
    "t_a1",
    "t_a2",
    "t_a3",
    "t_a4",
    "t_a5",
    "Interaction",
]

COLUMN_DSSP = COLUMNS_BIG[:-1]
