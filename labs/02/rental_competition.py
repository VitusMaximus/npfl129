#!/usr/bin/env python3

# f5419161-0138-4909-8252-ba9794a63e53
# 964bdfc8-60b0-4398-b837-7c2520532d17
# 4b50a6fb-a4a6-4b30-9879-0b671f941a72

import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request
import sklearn.compose
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import numpy as np
import numpy.typing as npt
import sklearn.pipeline

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")
parser.add_argument("--test_size", default=0.15, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")

class Dataset:
    """Rental Dataset.

    The dataset instances consist of the following 12 features:
    - season (1: winter, 2: spring, 3: summer, 4: autumn)
    - year (0: 2011, 1: 2012)
    - month (1-12)
    - hour (0-23)
    - holiday (binary indicator)
    - day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - working day (binary indicator; a day is neither weekend nor holiday)
    - weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - relative humidity (0-1 range)
    - windspeed (normalized to 0-1 range)

    The target variable is the number of rented bikes in the given hour.
    """
    def __init__(self,
                 name="rental_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)

def create_transformer(x_train):
    categorical_columns = np.argwhere(np.all(np.mod(x_train, 1) == 0, axis=0)).ravel() # Only integers
    continuous_columns = np.setdiff1d(np.arange(x_train.shape[1]), categorical_columns)
    preprocessor = sklearn.compose.ColumnTransformer(
        transformers=[
            ("encoder", sklearn.preprocessing.OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_columns),
            ("scaler", sklearn.preprocessing.StandardScaler(), continuous_columns)
        ]
    )
    polynomial_featurer = sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)
    pipe = sklearn.pipeline.Pipeline((("prep",preprocessor),("poly",polynomial_featurer)))
    pipe.fit(x_train)
    return pipe

def train_ridge(X_train,y_train, X_test, y_test):
    lambdas = np.geomspace(0.01, 10, num=500)
    best_rmse = float('inf')
    for l in lambdas:
        model = sklearn.linear_model.Ridge(alpha=l)
        model.fit(X_train,y_train)
        preds = model.predict(X_test)
        rmse = sklearn.metrics.mean_squared_error(preds,y_test,squared=False)
        if rmse < best_rmse:
            best_model = model
            best_rmse = rmse
    return best_model

def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(train.data, train.target, test_size=args.test_size, random_state=args.seed)
        # TODO: Train a model on the given dataset and store it in `model`.
        pipe = create_transformer(X_train)
        
        X_train = pipe.transform(X_train)
        X_test = pipe.transform(X_test)
        model = train_ridge(X_train,y_train,X_test,y_test)
        model = sklearn.pipeline.Pipeline((("pipe",pipe),("model",model)))
        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)
        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
