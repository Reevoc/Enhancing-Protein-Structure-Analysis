from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler


def normalization_angles(df, scale):
    if scale == "MinMaxScaler":
        scaler = MinMaxScaler()
    if scale == "StandardScaler":
        scaler = StandardScaler()
    column_angles = ["s_phi", "s_psi", "t_psi", "t_phi"]
    df[column_angles] = scaler.fit_transform(df[column_angles])
    return df


def normalization_achety(df, scale):
    if scale == "MinMaxScaler":
        scaler = MinMaxScaler()
    if scale == "StandardScaler":
        scaler = StandardScaler()
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


def normalization_rsa(df, scale):
    if scale == "MinMaxScaler":
        scaler = MinMaxScaler()
    if scale == "StandardScaler":
        scaler = StandardScaler()
    column_rsa = ["s_rsa", "t_rsa"]
    df[column_rsa] = scaler.fit_transform(df[column_rsa])
    return df


def normalization_half_sphere(df, scale):
    if scale == "MinMaxScaler":
        scaler = MinMaxScaler()
    if scale == "StandardScaler":
        scaler = StandardScaler()
    column_half_sphere = ["s_up", "s_down", "t_up", "t_down"]
    df[column_half_sphere] = scaler.fit_transform(df[column_half_sphere])
    return df


def normalization_category(df):
    column_category_ss8 = "s_ss8"
    column_category_ss3 = "s_ss3"
    column_category_resn = "s_resn"
    column_category_resnt = "t_resn"
    column_category_ss8t = "t_ss8"
    column_category_ss3t = "t_ss3"
    le = LabelEncoder()
    df[column_category_ss3] = le.fit_transform(df[column_category_ss3])
    df[column_category_resn] = le.fit_transform(df[column_category_resn])
    df[column_category_ss8] = le.fit_transform(df[column_category_ss8])
    df[column_category_ss3t] = le.fit_transform(df[column_category_ss3t])
    df[column_category_resnt] = le.fit_transform(df[column_category_resnt])
    df[column_category_ss8t] = le.fit_transform(df[column_category_ss8t])
    return df


def normalization_dssp(df, scale):
    if scale == "MinMaxScaler":
        scaler = MinMaxScaler()
    if scale == "StandardScaler":
        scaler = StandardScaler()
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
    return df


def all_normalization(df, scale):
    if scale != "no_normalization":
        df = normalization_angles(df, scale)
        df = normalization_achety(df, scale)
        df = normalization_rsa(df, scale)
        df = normalization_half_sphere(df, scale)
        df = normalization_dssp(df, scale)
    df = normalization_category(df)
    df = normalization_interactions(df)
    return df
