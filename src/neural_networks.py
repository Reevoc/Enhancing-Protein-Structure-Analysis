import numpy as np
from keras import Sequential
from keras.layers import BatchNormalization, Dense, Dropout
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, train_test_split

import configuration as conf
import src.split as split


def get_dim(X, Y):
    input_dim = X.shape[0]
    print("input_dim", input_dim)
    num_classes = len(Y[0])
    print("num_classes", num_classes)
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
    model.add(Dense(num_classes, activation="sigmoid"))

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
    model.add(Dense(num_classes, activation="sigmoid"))

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    return model


def create_model_3(input_dim, num_classes, optimizer="adam", dropout_rate=0.2):
    model = Sequential()
    model.add(Dense(896, activation="relu", input_dim=input_dim))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(num_classes, activation="sigmoid"))
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


def train(
    df,
    model_name,
    optimizer="adam",
    epochs=30,
    batch_size=conf.BATCH_SIZE,
    dropout_rate=0.2,
    kfold=conf.KFOLDS,
):
    print("Start training ")
    # tf_config.threading.set_inter_op_parallelism_threads(conf.MTHREAD)
    INPUT_DIM = 42
    NUM_CLASSES = 7
    model = get_model(model_name)(INPUT_DIM, NUM_CLASSES, optimizer, dropout_rate)
    X, Y = split.get_dataset(df)
    kf = KFold(n_splits=kfold, shuffle=True, random_state=1)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1, shuffle=True
    )

    model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True,
        # validation_data=(X_test, Y_test),
    )
    model.save(f"{model_name}.h5")


def kfold_train(
    df,
    model_name,
    optimizer="adam",
    epochs=30,
    batch_size=conf.BATCH_SIZE,
    dropout_rate=0.2,
    kfold=conf.KFOLDS,
):
    print("Start training ")
    # tf_config.threading.set_inter_op_parallelism_threads(conf.MTHREAD)
    INPUT_DIM = 42
    NUM_CLASSES = 7
    model = get_model(model_name)(INPUT_DIM, NUM_CLASSES, optimizer, dropout_rate)
    X, Y = split.get_dataset(df)
    kf = KFold(n_splits=kfold, shuffle=True, random_state=1)
    # X_train, X_test, Y_train, Y_test = train_test_split(
    #     X, Y, test_size=0.1, shuffle=True
    # )

    accuracy_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []

    # Perform cross-validation
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"\n\n\tStarting {i} cross fold validation")
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        model = get_model(model_name)(INPUT_DIM, NUM_CLASSES, optimizer, dropout_rate)
        # Train the model
        model.fit(
            X_train,
            Y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=True,
            validation_data=(X_test, Y_test),
        )

        y_pred = model.predict(X_test)

        threshold = 0.5
        y_pred = (y_pred >= threshold).astype(int)

        accuracy = accuracy_score(Y_test, y_pred)
        f1 = f1_score(
            Y_test, y_pred, average="micro"
        )  # micro-averaged F1 score for multilabel
        precision = precision_score(Y_test, y_pred, average="micro")
        recall = recall_score(Y_test, y_pred, average="micro")

        # Append the performance metrics for this fold to the lists
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

    # # Print the evaluation metrics for each fold
    # print("Fold Loss:", loss)
    # print("Fold Accuracy:", accuracy)


def gridsearch(df, model_name, f, param):
    optimizers, dropout_rate, epochs = param

    for dr in dropout_rate:
        for optimizer in optimizers:
            for epoch in epochs:
                print(f"Start training with {dr} {optimizer} {epoch}")
                metrics = kfold_train(
                    df,
                    model_name,
                    epochs=epoch,
                    optimizer=optimizer,
                    dropout_rate=dr,
                    kfold=conf.KFOLDS,
                )

                (
                    average_accuracy,
                    average_f1,
                    average_precision,
                    average_recall,
                ) = metrics

                f.write(
                    f"{model_name}\t{dr}\t{optimizer}\t{epochs}\t{average_accuracy}\t{average_f1}\t{average_precision}\t{average_recall}\n"
                )
