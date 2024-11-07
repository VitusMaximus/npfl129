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
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.compose
import sklearn.preprocessing
import sklearn.pipeline
import numpy as np
import numpy.typing as npt

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--log",default="logs5.txs",type=str)
parser.add_argument("--test_size", default=0.1, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="thyroid_competition5.model", type=str, help="Model path")


class Dataset:
    """Thyroid Dataset.

    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features

    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """
    def __init__(self,
                 name="thyroid_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)

def print_grid_res(model,args):
    f = None
    if args.log:
        f  = open(args.log,'w')
    for rank, accuracy, params in zip(model.cv_results_["rank_test_score"],
                                        model.cv_results_["mean_test_score"],
                                        model.cv_results_["params"]):
        print("Rank: {:2d} Cross-val: {:.1f}%".format(rank, 100 * accuracy),
                *("{}: {:<5}".format(key, value) for key, value in params.items()))
        if f != None:
            print("Rank: {:2d} Cross-val: {:.1f}%".format(rank, 100 * accuracy),
                *("{}: {:<5}".format(key, value) for key, value in params.items()),file=f)
    if f != None:
        f.close()
            


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        categ_c = np.arange(15)
        float_c = np.arange(15,21)
        #train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(train.data,train.target,test_size=args.test_size,random_state=args.seed)
        trans = sklearn.compose.ColumnTransformer(
            transformers=[
                ('categ',sklearn.pipeline.Pipeline((
                    ("encoder", sklearn.preprocessing.OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
                    ('spline',sklearn.preprocessing.SplineTransformer()))
                ), categ_c),
                ("scaler", sklearn.preprocessing.StandardScaler(), float_c)
            ]
        )   
        pipe = sklearn.pipeline.Pipeline((('prep',trans),
                                      ('poly',sklearn.preprocessing.PolynomialFeatures()),
                                      ('model',sklearn.linear_model.LogisticRegression(penalty='l2',random_state=args.seed,max_iter=1400,n_jobs=-1))))
        
        params = {
            'prep__categ__spline__n_knots': [2,3,4],
            'poly__degree': [1],
            'model__solver':['sag','lbfgs'],
            'model__C':[160,180,200,220,240,260,280,300],
            'model__penalty':['l2']    
        }
        f = sklearn.model_selection.StratifiedKFold()
        model = sklearn.model_selection.GridSearchCV(estimator=pipe,param_grid=params,cv=f,scoring='accuracy',n_jobs=-1)
        model.fit(train.data,train.target)
        print_grid_res(model,args)
        model = model.best_estimator_
        
        
        #pipe.fit(train_data,train_target)
        #test_pred = pipe.predict(test_data)
        #test_acc = np.mean((test_pred >= 0.5) == test_target)
        #print(test_acc)
        # Serialize the model.
        
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)
        #test = Dataset()
        """
        models = ["1.model","2.model","3.model"]
        m = []
        for mod in models:
            with lzma.open(mod, "rb") as model_file:
                m.append(pickle.load(model_file))

        # TODO: Generate `predictions` with the test set predictions.
        preds = [mod.predict(test.data) for mod in m]
        preds = np.array(preds)
        preds = np.sum(preds,0)
        predictions = np.where(preds >= 2,1,0)"""
        with lzma.open(args.model_path, "rb") as model_file:
                model =(pickle.load(model_file))
        
        predictions = model.predict(test.data)
        return predictions


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
