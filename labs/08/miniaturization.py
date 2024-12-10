#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt
import sklearn.neural_network
from scipy.ndimage import shift, rotate, zoom
parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--crop", default=3, type=int, help="PAD for random crop")
parser.add_argument("--rotate", default=15, type=int, help="Max rotation for random rotate")
parser.add_argument("--num_augment", default=16, type=int, help="Number of augmentations per img")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="miniaturization.model", type=str, help="Model path")
parser.add_argument("--test_size", default=500, type=int, help="Test set size")
SIZE:int = 28 # Mnist img size in each dim
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


# The following class modifies `MLPClassifier` to support full categorical distributions
# on input, i.e., each label should be a distribution over the predicted classes.
# During prediction, the most likely class is returned, but similarly to `MLPClassifier`,
# the `predict_proba` method returns the full distribution.
# Note that because we overwrite a private method, it is guaranteed to work only with
# scikit-learn 1.5.2, but it will most likely work with any 1.5.*.
class MLPFullDistributionClassifier(sklearn.neural_network.MLPClassifier):
    class FullDistributionLabels:
        y_type_ = "multiclass"

        def fit(self, y):
            return self

        def transform(self, y):
            return y

        def inverse_transform(self, y):
            return np.argmax(y, axis=-1)

    def _validate_input(self, X, y, incremental, reset):
        X, y = self._validate_data(X, y, multi_output=True, dtype=(np.float64, np.float32), reset=reset)
        if (not hasattr(self, "classes_")) or (not self.warm_start and not incremental):
            self._label_binarizer = self.FullDistributionLabels()
            self.classes_ = np.arange(y.shape[1])
        return X, y

def label_smoothing(labels:np.ndarray, smoothing = 0.1):
    return labels * (1-smoothing) + smoothing/10

def augment(img:np.ndarray, pad = 2, rotation = 15):
    img = img.reshape((SIZE,SIZE))
    padded = np.pad(img,((pad,pad),(pad,pad)),constant_values=0)
    if rotation != 0:
        rot = np.random.randint(-rotation,rotation+1)
        padded = rotate(padded,rot,reshape=False)
    cropx = np.random.randint(1,pad*2)
    cropy = np.random.randint(1,pad*2)
    return padded[cropx:cropx+SIZE,cropy:cropy+SIZE].flatten()

def create_croped(data:np.ndarray,args):
    cropped = np.zeros((data.shape[0],data.shape[1]))
    for i in range(data.shape[0]):
        cropped[i] = augment(data[i],args.crop,args.rotate)
    return cropped

def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        train.data, train.target, test_size=args.test_size, random_state=args.seed)
        
        for i in range(args.num_augment):    
            cropped = create_croped(train_data, args)
            # Works for crop One
            if i == 0:
                train_X = cropped
                train_Y = np.copy(train_target)
            else:
                train_X = np.concatenate((train_X,cropped),0)
                train_Y = np.concatenate((train_Y,train_target))
        train_Y = np.eye(10)[train_Y]
        # TODO: Train a model on the given dataset and store it in `model`.
        mlp = MLPFullDistributionClassifier((300, 200),verbose=True,learning_rate="invscaling", early_stopping=True, validation_fraction=0.1)
        mlp.fit(train_X,train_Y)
        print(mlp.score(test_data,test_target))
        # If you trained one or more MLPs, you can use the following code
        # to compress it significantly (approximately 12 times). The snippet
        # assumes the trained MLP is in the `mlp` variable.
        mlp._optimizer = None
        for i in range(len(mlp.coefs_)): mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
        for i in range(len(mlp.intercepts_)): mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(mlp, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)
        # TODO: Generate `predictions` with the test set predictions.
        predictions = model.predict(test.data)
        return predictions
    
def augment_image(image):
    angle = np.random.uniform(-20, 20)  # Rotace
    rotated = rotate(image, angle, reshape=False, mode='nearest')

    shift_x = np.random.uniform(-3, 3)  # Posun
    shift_y = np.random.uniform(-3, 3)
    shifted = shift(rotated, shift=(shift_y, shift_x), mode='nearest')

    zoom_factor = np.random.uniform(1.0, 1.2)  # Zoom
    zoomed = zoom(shifted, zoom=zoom_factor, mode='nearest')
    
    # Oříznutí nebo doplnění na 28x28
    center_crop = zoomed[
        max(0, (zoomed.shape[0] - 28) // 2):max(0, (zoomed.shape[0] - 28) // 2) + 28,
        max(0, (zoomed.shape[1] - 28) // 2):max(0, (zoomed.shape[1] - 28) // 2) + 28
    ]
    if center_crop.shape != (28, 28):
        result = np.zeros((28, 28))
        result[:center_crop.shape[0], :center_crop.shape[1]] = center_crop
        return result
    return center_crop


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
