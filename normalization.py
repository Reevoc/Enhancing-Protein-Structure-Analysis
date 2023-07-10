from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def normalization_angles(df):
    column_angles = ["s_phi", "s_psi", "t_psi", "t_phi"]
    scaler = MinMaxScaler()
    df[column_angles] = scaler.fit_transform(df[column_angles])
    return df


def normalization_achety(df):
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
    scaler = MinMaxScaler()
    df[column_achety] = scaler.fit_transform(df[column_achety])
    return df


def normalization_rsa(df):
    column_rsa = ["s_rsa", "t_rsa"]
    scaler = MinMaxScaler()
    df[column_rsa] = scaler.fit_transform(df[column_rsa])
    return df


def normalization_half_sphere(df):
    column_half_sphere = ["s_up", "s_down", "t_up", "t_down"]
    scaler = MinMaxScaler()
    df[column_half_sphere] = scaler.fit_transform(df[column_half_sphere])
    return df


def normalization_category(df):
    column_category_ss8 = ["s_ss8", "t_ss8"]
    column_category_ss3 = ["s_ss3", "t_ss3"]
    column_category_resn = ["s_resn", "t_resn"]
    le = LabelEncoder()
    df[column_category_ss8] = le.fit_transform(df[column_category_ss8])
    df[column_category_ss3] = le.fit_transform(df[column_category_ss3])
    df[column_category_resn] = le.fit_transform(df[column_category_resn])
    return df


def all_normalization(df):
    df = normalization_angles(df)
    df = normalization_achety(df)
    df = normalization_rsa(df)
    df = normalization_half_sphere(df)
    df = normalization_category(df)
    return df
