from keras import Sequential
from keras.layers import BatchNormalization, Dense, Dropout
from keras.utils import to_categorical

import split


def creating_dataset_for_train(df):
    X, Y = split.split_dataset(df)
    X = X.astype(float)
    Y = to_categorical(Y)
    # split the dataset into train and test
    X_train, X_test, Y_train, Y_test = split.split_train_test(X, Y)

    return X_train, X_test, Y_train, Y_test


def create_input_dim(X_train, Y_train):
    input_dim = X_train.shape[1]
    print(input_dim)
    num_classes = Y_train.shape[1]
    print(num_classes)
    return input_dim, num_classes


def create_model_1(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(128, activation="relu", input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def create_model_2(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(128, activation="relu", input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(512, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    return model


def create_model_3(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(896, activation="relu", input_dim=input_dim))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def evaluate_model(model, X_test, Y_test):
    loss, accuracy = model.evaluate(X_test, Y_test)
    return loss, accuracy


def get_model(name):
    if name == "model_1":
        return create_model_1
    if name == "model_2":
        return create_model_2
    if name == "model_3":
        return create_model_3


def model(df, model_name):
    X_train, X_test, Y_train, Y_test = creating_dataset_for_train(df)
    input_dim, num_classes = create_input_dim(X_train, Y_train)
    model = get_model(model_name)(input_dim, num_classes)
    model.fit(
        X_train,
        Y_train,
        epochs=10,
        batch_size=128,
        verbose=True,
        validation_data=(X_test, Y_test),
    )
    loss, accuracy = evaluate_model(model, X_test, Y_test)
    return loss, accuracy
