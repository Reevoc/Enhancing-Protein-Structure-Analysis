
The code you've provided currently implements a train-test split approach to validate the performance of a model. To modify this code to use K-Fold Cross Validation (CV), we'll make use of the `KerasClassifier` wrapper and `cross_val_score` function from Scikit-Learn. K-Fold CV will provide a more robust estimate of model performance by splitting the data into 'K' parts, or folds, and performing 'K' iterations of training and validation, each time with a different fold used as the validation set.

Here's the fixed code:

```python
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, KFold
from keras.utils import to_categorical
import numpy as np

def creating_dataset(df):
    X, Y = split.split_dataset(df)
    X = X.astype(float)
    Y = to_categorical(Y)
    return X, Y

def create_input_dim(X, Y):
    input_dim = X.shape[1]
    print(input_dim)
    num_classes = Y.shape[1]
    print(num_classes)
    return input_dim, num_classes

# ... Model creation functions remain the same ...

def get_model(name):
    if name == "model_1":
        return create_model_1
    if name == "model_2":
        return create_model_2
    if name == "model_3":
        return create_model_3

def model(df, model_name):
    X, Y = creating_dataset(df)
    input_dim, num_classes = create_input_dim(X, Y)

    # Define a wrapper for our model
    model_wrapper = KerasClassifier(build_fn=get_model(model_name), input_dim=input_dim, num_classes=num_classes, epochs=10, batch_size=128, verbose=1)

    # Perform k-fold cross-validation
    kfold = KFold(n_splits=10, shuffle=True)
    results = cross_val_score(model_wrapper, X, np.argmax(Y, axis=1), cv=kfold)
    
    return results.mean(), results.std()
```

This version of the function `model` performs 10-fold cross-validation on the input data and returns the mean and standard deviation of the model's accuracy over the 10 folds.

Note that we use `np.argmax(Y, axis=1)` to convert the one-hot encoded labels back to integer format, because `cross_val_score` expects integer labels.

Also note that the batch size, number of epochs, and verbosity are hardcoded into the `KerasClassifier` wrapper. You might want to adjust these or make them parameters of your `model` function, depending on your needs.

Please replace the `split.split_dataset` function with the appropriate function to split your data into features and target. I've assumed that it's a function that splits your DataFrame into a features matrix X and a target matrix Y.
