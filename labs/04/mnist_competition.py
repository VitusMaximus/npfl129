#!/usr/bin/env python3
# f5419161-0138-4909-8252-ba9794a63e53
# 964bdfc8-60b0-4398-b837-7c2520532d17
# 4b50a6fb-a4a6-4b30-9879-0b671f941a72

import warnings
warnings.filterwarnings('ignore')

import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import sklearn.datasets
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import numpy.typing as npt

#from sklearn import neighbors
#from sklearn import svm
#from sklearn.svm import SVC
#from sklearn import tree
#from sklearn.model_selection import GridSearchCV


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
parser.add_argument("--hidden_layers", nargs=1, type=int, default=[500], help="Sizes of hidden layers")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="mnist_competition.model", type=str, help="Model path")

class Dataset:
    """MNIST Dataset.

    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in the range 0-255, the targets are numbers 0-9.
    """
    def __init__(self,
                 name="mnist.train.npz",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset, i.e., `data` and optionally `target`.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28*28]).astype(float)

# Method for visualisation of greyscale maps of handwritten images. (Only for me)
def create_image_from_array(array):
    import numpy as np
    import matplotlib.pyplot as plt
    if array.size == 28 * 28:
        array = array.reshape(28, 28)
    else:
        raise ValueError("Input array must be a 28x28 pixel array")

    # Display the image with gray colormap and no axis
    plt.imshow(array, cmap='gray')
    plt.axis('off')  # Hide the axes for better visualization
    plt.show()

def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
            train.data, train.target, test_size=args.test_size, random_state=args.seed
        )
        
        """
        #SVC implementation with gridsearch.
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        print("aspon tu som")
        svc = SVC()
        param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['rbf', 'linear']}
        grid_search = GridSearchCV(svc, param_grid, cv=5)

        grid_search.fit(X_train, y_train)
        """

        """
        #SVM implementation
        model = svm.SVC(gamma=0.001) 
        model.fit(X_train, y_train)

        """

        """
        #Decision Tree implementation

        class_tree=tree.DecisionTreeClassifier()
        class_tree.fit(X_train, y_train)
        """

        #input data normalization
        scaler = StandardScaler()

        data_scaled = scaler.fit_transform(train.data)

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_val)

        model = MLPClassifier(
            hidden_layer_sizes=tuple(args.hidden_layers),
            random_state=args.seed
        )
        
        #model.fit(X_train_scaled,y_train)

        # Predict on the validation set
        #y_pred = model.predict(X_test_scaled)
        # Print the accuracy on the validation set
        #print("Accuracy of model on validation data = %2f%%" % (accuracy_score(y_val, y_pred) * 100))
        
        # nafitovat na celu mnozinu
        model.fit(data_scaled,train.target)
        
        #vypis pre info:

        #y_pred = model.predict(data_scaled)
        #print("Accuracy of model on validation data = %2f%%" % (accuracy_score(train.target, y_pred) * 100))
    

        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

        
    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        # po pripade spravit pipelinou scaling

        scaler = StandardScaler()

        test_data_scaled =scaler.fit_transform(test.data)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = model.predict(test_data_scaled)

        # Vypis pre info:
        #print("Accuracy of model = %2f%%" % (accuracy_score(test.target, predictions )*100))
        
        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
