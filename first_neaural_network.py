import data_manipulation
import split
import normalization
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MultiLabelBinarizer


def split_train_test(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X.values, Y.values, test_size=0.2, random_state=42
    )
    return X_train, X_test, Y_train, Y_test

def create_model(input_dim, num_classes):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_dim=input_dim),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model, X_test, Y_test):
    loss, accuracy = model.evaluate(X_test, Y_test)
    print("Accuracy", accuracy)


if __name__ == "__main__":
    # Load the dataset
    df = data_manipulation.generate_interaction_dataframe()
    # normalize the dataset
    df = normalization.all_normalization(df)
    # split the dataset
    X, Y = split.split_dataset(df)
    # split the dataset into train and test
    X_train, X_test, Y_train, Y_test = split_train_test(X, Y)

    mlb = MultiLabelBinarizer()
    Y_train = mlb.fit_transform(Y_train)
    Y_test = mlb.transform(Y_test)

    # Create the model
    input_dim = X_train.shape[1]
    num_classes = Y_train.shape[1]
    model = create_model(input_dim, num_classes)

    # Train the model
    model.fit(X_train, Y_train, epochs=10, batch_size=32)

    evaluate_model(model, X_test, Y_test)