#!/usr/bin/env python3``
# f5419161-0138-4909-8252-ba9794a63e53
# 964bdfc8-60b0-4398-b837-7c2520532d17
# 4b50a6fb-a4a6-4b30-9879-0b671f941a72
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)
    
    # Load the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Append a constant feature with value 1 to the end of every input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    data = np.pad(data, [(0, 0), (0, 1)], constant_values=1)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights.
    weights = generator.uniform(size=[train_data.shape[1], args.classes], low=-0.1, high=0.1)

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        #
        # Note that you need to be careful when computing softmax because the exponentiation
        # in softmax can easily overflow. To avoid it, you should use the fact that
        # $softmax(z) = softmax(z + any_constant)$ and compute $softmax(z) = softmax(z - maximum_of_z)$.
        # That way we only exponentiate non-positive values, and overflow does not occur.

        permutation_X = train_data[permutation]
        permutation_Y = train_target[permutation]

        for start in range(0, len(train_data), args.batch_size):

            X = permutation_X[start : start + args.batch_size]
            Y = permutation_Y[start : start + args.batch_size]

            logits = X @ weights
            predictions = softmax(logits)

            #loss = cross_entropy_loss(predictions, Y)

            gradient = compute_gradient(X, Y, predictions)
            weights -= args.learning_rate * gradient


        # TODO: After the SGD epoch, measure the average loss and accuracy for both the
        # train test and the test set. The loss is the average MLE loss (i.e., the
        # negative log-likelihood, or cross-entropy loss, or KL loss) per example.
        train_pred = softmax(train_data @ weights)
        test_pred = softmax(test_data @ weights)

        train_accuracy = compute_accuracy(train_pred, train_target)
        train_loss = cross_entropy_loss(train_pred, train_target)

        test_accuracy = compute_accuracy(test_pred, test_target)
        test_loss = cross_entropy_loss(test_pred, test_target)


        #train_accuracy, train_loss, test_accuracy, test_loss = ...

        print("After epoch {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            epoch + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

    return weights, [(train_loss, 100 * train_accuracy), (test_loss, 100 * test_accuracy)]


def softmax(logits : np.ndarray):
    logits -= np.max(logits, axis = 1, keepdims=True)

    sum = np.sum(np.exp(logits), axis = 1, keepdims=True)
    
    return (np.exp(logits)) / sum


def cross_entropy_loss(predictions, targets):

    true_pred = predictions[range(predictions.shape[0]), targets]
    true_log_probs = - np.log(true_pred)
    return np.mean(true_log_probs)

def compute_gradient(X, y_true, predictions):
    n_samples = predictions.shape[0]
    predictions[range(n_samples), y_true] -= 1  # predictions - y_true (one_hot)
    return X.T @ predictions / n_samples

def compute_accuracy(predictions, y_true):
    sum =  0
    n_samples = predictions.shape[0]
    for i in range(n_samples):
        if np.argmax(predictions[i]) == y_true[i]:
            sum += 1
    return sum / n_samples

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(main_args)
    print("Learned weights:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")
