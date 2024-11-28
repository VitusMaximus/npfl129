#!/usr/bin/env python3
import os
import sys
import urllib.request
import argparse
import lzma
import pickle
from typing import Optional
from sklearn import neural_network
import sklearn.model_selection
from sklearn.preprocessing import OneHotEncoder
import sklearn

import numpy as np
import re


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization2.model", type=str, help="Model path")
parser.add_argument("--window_size", default=5, type=int)


class Dictionary:
    def __init__(self,
                 name="fiction-dictionary.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dictionary to `variants`
        self.variants = {}
        with open(name, "r", encoding="utf-8-sig") as dictionary_file:
            for line in dictionary_file:
                nodia_word, *variants = line.rstrip("\n").split()
                self.variants[nodia_word] = variants


class Dataset:
    PAD = "<pad>"
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"
    #ALL_SYMBOLS = [PAD, '\n', ' ', '!', '"', "'", '(', ')', '/', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    ALL_SYMBOLS = [PAD, '\n', ' ', ',', '.', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'infrequent']

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)
        self.data_list = [char for char in self.data]
        self.windows = []
        self.window_targets = []

    def create_windows(self, winodw_size):
        padded_data = np.array([self.PAD]*winodw_size + self.data_list + [self.PAD]*winodw_size)     #Padding
        #padded_data = np.array([chr(x) for x in padded_data_enc])
        encoder = OneHotEncoder(sparse_output=False, categories=[self.ALL_SYMBOLS], handle_unknown="infrequent_if_exist")
        padded_data_OneHot = encoder.fit_transform(padded_data.reshape(-1,1))
        #letters = {}
        for i in range(len(self.data)):
            
            if self.data[i] in self.LETTERS_NODIA:
                self.windows.append(padded_data_OneHot[i : i+2*winodw_size + 1])
                self.window_targets.append(self.LETTERS_DIA.find(self.target[i]))  # Returns index of letter with diacritic or -1
            #letters[self.data[i]] = True
        #print(sorted(letters.keys()))


def compute_accuracy(predictions, targets):
    sum = 0
    for i in range(len(predictions)):
        if predictions[i] == targets[i]:
            sum += 1

    return sum/len(predictions)

def compute_accuracy_words(translation, original):
    translation = translation.split(" ")
    original = original.split(" ")

    sum = 0
    for i in range(len(original)):
        if original[i] == translation[i]:
            sum += 1
    
    return sum/len(original)

def translate1(predictions, data):
    result = []
    j = 0
    for i in range(len(data)):
        result.append(data[i])
        if data[i] in Dataset.LETTERS_NODIA:
            if predictions[j] != -1:
                result[-1] = Dataset.LETTERS_DIA[predictions[j]]
            j += 1
    return ''.join(result)


def map_predictions(pred, data):
    mapping = np.zeros(len(data))
    j = 0
    for i in range(len(data)):
        if data[i].lower() in Dataset.LETTERS_NODIA:
            mapping[i] == pred[j]         #j-th prediction is for i-th character
            j += 1
        else:
            mapping[i] == -1
    return mapping

word_delimiters = [" ", "_", "-", ".", ",", "?", "!", ":", "\n", "\t"]

def split_pred(pred, data):
    split_pred = []
    split_words = []
    j = 0
    for i in range(len(data)):
        word_pred = []
        if data[i].lower() in Dataset.LETTERS_NODIA:
            word_pred.append(pred[j])
            j += 1
        if data[i] in word_delimiters:
            pass
def dia_char(char, prediction):
    pass

def translate_word(word, predict_proba, posibilities):
    res = []
    for i in range(len(word)):
        res.append(dia_char(word[i], np.argmax(predict_proba[i])))
    dia_word = "".join(res)
    if posibilities == None: return dia_word
    if dia_word.strip() in posibilities: return dia_word




def translate(dict, pred, data):
    res = []
    mapping = map_predictions(pred, data)
    i = 0
    for word in re.split(r'(\s,.?!-_:)', data):
        dia_word = translate_word(word, mapping[i:i+len(word)], dict)
        i += len(word)
        res.append(dia_word)





    str.split()

def main(args: argparse.Namespace) -> Optional[str]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        train.create_windows(args.window_size)
        windows = np.array(train.windows)
        window_targets = np.array(train.window_targets)
        #windows_OneHot = train.windows_to_OneHot()
        #print([[chr(x) for x in window] for window in train.windows])
        #print([train.window_targets])
        windows_flat = windows.reshape(windows.shape[0], -1)

        X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(windows_flat, window_targets, test_size=0.1, random_state=args.seed)
        # TODO: Train a model on the given dataset and store it in `model`.
        model = neural_network.MLPClassifier((400), verbose=True, max_iter=12, early_stopping=False, learning_rate="invscaling" )
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        accuracy = compute_accuracy(predictions, y_val)
        print(accuracy)
        #text = train.data
        translation = translate(predictions, train.data)
        print(translation[:300])

        accuracy_words = compute_accuracy_words(translation, train.target)
        print(accuracy_words)
        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)
        test.create_windows(args.window_size)

        windows = np.array(test.windows)
        windows_flat = windows.reshape(windows.shape[0], -1)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions. Specifically,
        # produce a diacritized `str` with exactly the same number of words as `test.data`.
        predictions = model.predict(windows_flat)
        translation = translate(predictions, test.data)

        #print(compute_accuracy(predictions, test.data))

        return translation


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
