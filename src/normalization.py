from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# path append parent directory
import sys
import configuration as conf

# ------------------------------------------------------------------------
# SETUP FIXED VALUES FOR NORMALIZATION -----------------------------------
# ------------------------------------------------------------------------

STRUCTURE_TYPE = [
    "H",  # α-helix
    "B",  # residue in isolated β-bridge
    "E",  # extended strand, participates in β ladder
    "G",  # 3-helix (310 helix)
    "I",  # 5 helix (π-helix)
    "T",  # hydrogen bonded turn
    "S",  # bend
    "L",  # "-", "C", "L" - loop coil or irregular ?? (autocompleted)
    "-",
]

residue_letters = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "E",
    "Q",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
    "X",
]

atchley_scale = []
with open(conf.ATCHLEY_FILE) as f:
    next(f)
    for line in f:
        line = line.strip().split("\t")
        atchley_scale.append(line[1:])
    # transpose atchley_scale -> array of features and not residues
    atchley_scale = list(map(list, zip(*atchley_scale)))
    atchley_scale = [[float(x) for x in y] for y in atchley_scale]

# ------------------------------------------------------------------------
# NORMALIZATION FUNCTIONS ------------------------------------------------
# ------------------------------------------------------------------------


def normalization_angles(df, scaler):
    print("\tNormalization angles")
    column_angles = ["s_phi", "s_psi", "t_psi", "t_phi"]
    df[column_angles] = scaler.fit_transform(df[column_angles])
    return df


def normalization_achety(df, scaler):
    print("\tNormalization achety")
    column_achety = [
        "s_a1",
        "s_a2",
        "s_a3",
        "s_a4",
        "s_a5",
        "t_a1",
        "t_a2",
        "t_a3",
        "t_a4",
        "t_a5",
    ]
    df[column_achety] = scaler.fit_transform(df[column_achety])
    return df


def normalization_rsa(df, scaler):
    print("\tNormalization rsa")
    column_rsa = ["s_rsa", "t_rsa"]
    df[column_rsa] = scaler.fit_transform(df[column_rsa])
    return df


def normalization_half_sphere(df, scaler):
    print("\tNormalization half sphere")
    column_half_sphere = ["s_up", "s_down", "t_up", "t_down"]
    df[column_half_sphere] = scaler.fit_transform(df[column_half_sphere])
    return df


def normalization_category(df):
    print("\tNormalization category")
    column_category = [
        "s_ch",
        "t_ch",
        "s_ss3",
        "t_ss3",
        "s_ss8",
        "t_ss8",
        "s_resn",
        "t_resn",
    ]
    le = LabelEncoder()
    for column in column_category:
        df[column] = le.fit_transform(df[column])
    scaler = StandardScaler()
    df[column_category] = scaler.fit_transform(df[column_category])
    return df


def normalization_dssp(df, scaler):
    print("\tNormalization dssp")
    column_dssp_energy = [
        "s_nh_energy",
        "s_o_energy",
        "s_nh2_energy",
        "s_o2_energy",
        "t_nh_energy",
        "t_o_energy",
        "t_nh2_energy",
        "t_o2_energy",
    ]
    column_dssp_relidix = [
        "s_nh_relidix",
        "s_o_relidx",
        "s_nh2_relidx",
        "s_o2_relidx",
        "t_nh_relidix",
        "t_o_relidx",
        "t_nh2_relidx",
        "t_o2_relidx",
    ]
    df[column_dssp_energy] = scaler.fit_transform(df[column_dssp_energy])
    df[column_dssp_relidix] = scaler.fit_transform(df[column_dssp_relidix])
    return df


def normalization_interactions(df):
    df["Interaction"] = df["Interaction"].apply(
        lambda x: x[0] if isinstance(x, list) else x
    )
    label_encoder = LabelEncoder()
    df["Interaction"] = label_encoder.fit_transform(df["Interaction"])

    # print("\tNormalization interactions")
    # interactions = list(df["Interaction"])
    # interactions = map(
    #     lambda x: [
    #         1 if x and i in x else 0 for i in conf.INTERACTION_TYPES
    #     ],  # FIXME there are some None
    #     interactions,
    # )
    # df["Interaction"] = list(interactions)
    return df


def normalization_df(df, scale):
    print(f"normalization df {scale}")
    if scale != "no_normalization":
        if scale == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif scale == "StandardScaler":
            scaler = StandardScaler()
        else:
            raise ValueError("Invalid scale value")

        df = normalization_angles(df, scaler)
        df = normalization_achety(df, scaler)
        df = normalization_rsa(df, scaler)
        df = normalization_half_sphere(df, scaler)
        df = normalization_dssp(df, scaler)

    df = normalization_category(df)
    df = normalization_interactions(df)
    return df
