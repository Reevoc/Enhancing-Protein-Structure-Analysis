import numpy as np
from keras import Sequential
from keras.layers import BatchNormalization, Dense, Dropout
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold

import configuration as conf
import src.split as split


def get_dim(X, Y):
    input_dim = X.shape[1]
    num_classes = Y.shape[1]
    return input_dim, num_classes


def create_model_1(input_dim, num_classes, optimizer="adam", dropout_rate=0.2):
    model = Sequential()
    model.add(Dense(128, activation="relu", input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    return model


def create_model_2(input_dim, num_classes, optimizer="adam", dropout_rate=0.2):
    model = Sequential()
    model.add(Dense(128, activation="relu", input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(512, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    return model


def create_model_3(input_dim, num_classes, optimizer="adam", dropout_rate=0.2):
    model = Sequential()
    model.add(Dense(896, activation="relu", input_dim=input_dim))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    return model


def evaluate_model(model, X_test, Y_test):
    loss, accuracy = model.evaluate(X_test, Y_test)
    print("Loss", loss, "Accuracy", accuracy)
    return loss, accuracy


def get_model(name):
    if name == "model_1":
        return create_model_1
    if name == "model_2":
        return create_model_2
    if name == "model_3":
        return create_model_3


def train_test_indices(dim, percentage):
    indices = np.arange(dim)
    np.random.shuffle(indices)
    split = int((1 - percentage) * dim)
    return indices[:split], indices[split:]


def train(
    df,
    model_name,
    optimizer="adam",
    epochs=30,
    batch_size=conf.BATCH_SIZE,
    dropout_rate=0.2,
    balanced=False,
    f=None,
    train_index=None,
    test_index=None,
    percent4test=0.2,
    return_model=False,
):
    print("Start training ")
    X, Y = split.get_dataset(df, balanced=balanced)
    INPUT_DIM, NUM_CLASSES = get_dim(X, Y)
    model = get_model(model_name)(INPUT_DIM, NUM_CLASSES, optimizer, dropout_rate)

    if train_index is None or test_index is None:
        train_index, test_index = train_test_indices(X.shape[0], percent4test)

    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True,
        validation_data=(X_test, Y_test),
    )

    # model.save(f"{model_name}.h5")
    # model.evaluate(X_test, Y_test)

    Y_pred = model.predict(X_test)
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_test = np.argmax(Y_test, axis=1)

    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average="micro")
    f1 = f1_score(Y_test, Y_pred, average="micro")
    recall = recall_score(Y_test, Y_pred, average="micro")
    print(
        "accuracy ",
        accuracy,
        " f1_score ",
        f1,
        " recall ",
        recall,
        " precision ",
        precision,
    )
    if f:
        f.write(f"{accuracy}\t{f1}\t{precision}\t{recall}\n")

    if return_model:
        return accuracy, f1, precision, recall, model
    return accuracy, f1, precision, recall


def kfold_train(
    df,
    model_name,
    optimizer="adam",
    epochs=30,
    batch_size=conf.BATCH_SIZE,
    dropout_rate=0.2,
    kfold=conf.KFOLDS,
    balanced=False,
    f=None,
):
    # print("Start training ")
    # X, Y = split.get_dataset(df, balanced=balanced)
    # INPUT_DIM, NUM_CLASSES = get_dim(X, Y)
    # model = get_model(model_name)(INPUT_DIM, NUM_CLASSES, optimizer, dropout_rate)
    kf = KFold(n_splits=kfold, shuffle=True, random_state=1)

    accuracy_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []

    # Perform cross-validation
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        accuracy, f1, precision, recall = train(
            df,
            model_name,
            optimizer,
            epochs,
            batch_size,
            dropout_rate,
            balanced=balanced,
            train_index=train_index,
            test_index=test_index,
        )

        accuracy_scores.append(accuracy)
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

    # Step 6: Calculate Average Performance
    average_accuracy = np.mean(accuracy_scores)
    average_f1 = np.mean(f1_scores)
    average_precision = np.mean(precision_scores)
    average_recall = np.mean(recall_scores)

    # Print the average performance metrics
    print(f"Average Accuracy: {average_accuracy}")
    print(f"Average F1 Score: {average_f1}")
    print(f"Average Precision: {average_precision}")
    print(f"Average Recall: {average_recall}")
    return average_accuracy, average_f1, average_precision, average_recall


def predict(df, model: Sequential):  # FIXME
    X, Y = split.get_dataset(df)

    train_index, test_index = train_test_indices(X.shape[0], 0.5)

    _, X_test = X[train_index], X[test_index]
    _, Y_test = Y[train_index], Y[test_index]
    y_pred = model.predict(X_test)
    # Labels
    Y_test = np.argmax(Y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    print("accuracy", accuracy_score(Y_test, y_pred))
    print("f1_score", f1_score(Y_test, y_pred, average="micro"))
    print("recall", recall_score(Y_test, y_pred, average="micro"))
    print("precision", precision_score(Y_test, y_pred, average="micro"))

    # # # One Hot Encoding
    # # threshold = 0.5
    # # y_pred = (y_pred >= threshold).astype(int)
    # count_t = [0, 0, 0, 0, 0, 0, 0]
    # count_c = [0, 0, 0, 0, 0, 0, 0]
    # count_w = [0, 0, 0, 0, 0, 0, 0]
    # print(conf.INTERACTION_TYPES)
    # for i, j in zip(Y_test, y_pred):
    #     for k in range(get_dim(X, Y)[1]):
    #         if i[k] == 1:
    #             count_t[k] += 1
    #         if j[k] == 1:
    #             if i[k] == 1:
    #                 count_c[k] += 1
    #             else:
    #                 count_w[k] += 1
    # print("Accuracy", accuracy_score(Y_test, y_pred))
    # print(f"count_t\t{count_t}\ncount_c\t{count_c}\ncount_w\t{count_w}")