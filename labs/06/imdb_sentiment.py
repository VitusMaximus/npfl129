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

import numpy as np
import numpy.typing as npt
import sklearn.feature_extraction
import sklearn.metrics
import sklearn.model_selection
import sklearn.neural_network
import re

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="imdb_sentiment.model", type=str, help="Model path")
# TODO: Add other arguments (typically hyperparameters) as you need.


class Dataset:
    """IMDB dataset.

    This is a modified IMDB dataset for sentiment classification. The text is
    already tokenized and partially normalized.
    """
    def __init__(self,
                 name="imdb_train.txt",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and split it into `data` and `target`.
        self.data = []
        self.target = []
        with open(name) as f_imdb:
            for line in f_imdb:
                label, text = line.split("\t", 1)
                self.data.append(text)
                self.target.append(int(label))


def load_word_embeddings(
        name="imdb_embeddings.npz",
        url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
    """Load word embeddings.

    These are selected word embeddings from FastText. For faster download, it
    only contains words that are in the IMDB dataset.
    """
    if not os.path.exists(name):
        print("Downloading embeddings {}...".format(name), file=sys.stderr)
        urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
        os.rename("{}.tmp".format(name), name)

    with open(name, "rb") as f_emb:
        data = np.load(f_emb)
        words = data["words"]
        vectors = data["vectors"]
    embeddings = {word: vector for word, vector in zip(words, vectors)}
    return embeddings


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    word_embeddings = load_word_embeddings()

    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        print("Preprocessing dataset.", file=sys.stderr)
        # TODO: Preprocess the text such that you have a single vector per movie
        # review. You can experiment with different ways of pooling the word
        # embeddings: averaging, max pooling, etc. You can also try to exclude
        # words that do not contribute much to the meaning of the sentence (stop
        # words). See `sklearn.feature_extraction._stop_words.ENGLISH_STOP_WORDS`.
        train_as_vectors = text_to_vectors(train.data, word_embeddings)
            #spl = np.array_split(np.array(vectors), 10)
            #features = np.mean(np.array_split(np.array(vectors), 10), axis=1)

        

        train_x, validation_x, train_y, validation_y = sklearn.model_selection.train_test_split(
            train_as_vectors, train.target, test_size=0.25, random_state=args.seed)

        print("Training.", file=sys.stderr)
        # TODO: Train a model of your choice on the given data.
        model = sklearn.neural_network.MLPClassifier((300, 300), verbose=True, max_iter=26)
        model.fit(train_x, train_y)

        print("Evaluation.", file=sys.stderr)
        validation_predictions = model.predict(validation_x)
        validation_accuracy = sklearn.metrics.accuracy_score(validation_y, validation_predictions)
        print("Validation accuracy {:.2f}%".format(100 * validation_accuracy))

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Start by preprocessing the test data, ideally using the same
        # code as during training.
        test_as_vectors = text_to_vectors(test.data, word_embeddings)

        # TODO: Generate `predictions` with the test set predictions, either
        # as a Python list or a NumPy array.
        predictions = model.predict(test_as_vectors)

        return predictions




def text_to_vectors(data, word_embeddings):
    X_as_vectors = np.zeros((len(data), 300))
    for i, text in enumerate(data):
        split = text.split()
        vectors = [] 
        for word in split:
            if word in word_embeddings.keys():
                vectors.append(word_embeddings[word])
        X_as_vectors[i, :] = np.mean(vectors, axis=0)
    return X_as_vectors




if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)

