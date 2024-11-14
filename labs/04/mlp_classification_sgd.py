#!/usr/bin/env python3
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
parser.add_argument("--hidden_layer", default=50, type=int, help="Hidden layer size")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[tuple[np.ndarray, ...], list[float]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Load the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights.
    weights = [generator.uniform(size=[train_data.shape[1], args.hidden_layer], low=-0.1, high=0.1),
               generator.uniform(size=[args.hidden_layer, args.classes], low=-0.1, high=0.1)]
    biases = [np.zeros(args.hidden_layer), np.zeros(args.classes)]

    def ReLU(X):
        return np.maximum(0, X)
    
    def softmax(X):
        X -= np.max(X, axis = 1, keepdims=True)

        sum = np.sum(np.exp(X), axis = 1, keepdims=True)
    
        return (np.exp(X)) / sum



    def forward(inputs):
        # TODO: Implement forward propagation, returning *both* the value of the hidden
        # layer and the value of the output layer.
        #
        # We assume a neural network with a single hidden layer of size `args.hidden_layer`
        # and ReLU activation, where $ReLU(x) = max(x, 0)$, and an output layer with softmax
        # activation.
        #
        # The value of the hidden layer is computed as `ReLU(inputs @ weights[0] + biases[0])`.
        # The value of the output layer is computed as `softmax(hidden_layer @ weights[1] + biases[1])`.
        #
        # Note that you need to be careful when computing softmax, because the exponentiation
        # in softmax can easily overflow. To avoid it, you should use the fact that
        # $softmax(z) = softmax(z + any_constant)$ and compute $softmax(z) = softmax(z - maximum_of_z)$.
        # That way we only exponentiate values which are non-positive, and overflow does not occur.

        hidden_layer = ReLU(inputs @ weights[0] + biases[0])
        output_layer = softmax(hidden_layer @ weights[1] + biases[1])

        return hidden_layer, output_layer
    

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        permutation_X = train_data[permutation]
        permutation_Y = train_target[permutation]
        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        #
        # The gradient used in SGD has now four parts, gradient of `weights[0]` and `weights[1]`
        # and gradient of `biases[0]` and `biases[1]`.
        #
        # You can either compute the gradient directly from the neural network formula,
        # i.e., as a gradient of $-log P(target | data)$, or you can compute
        # it step by step using the chain rule of derivatives, in the following order:
        # - compute the derivative of the loss with respect to *inputs* of the
        #   softmax on the last layer,
        # - compute the derivative with respect to `weights[1]` and `biases[1]`,
        # - compute the derivative with respect to the hidden layer output,
        # - compute the derivative with respect to the hidden layer input,
        # - compute the derivative with respect to `weights[0]` and `biases[0]`.

        for start in range(0, len(train_data), args.batch_size):
            X = permutation_X[start : start + args.batch_size]
            Y = permutation_Y[start : start + args.batch_size]

            hidden, predictions = forward(X)

            W0, b0, W1, b1 = compute_gradients(X, hidden, predictions, Y, weights)

            weights[0] -= args.learning_rate * W0
            weights[1] -= args.learning_rate * W1
            biases[0] -= args.learning_rate * b0
            biases[1] -= args.learning_rate * b1


        # TODO: After the SGD epoch, measure the accuracy for both the
        # train test and the test set.
        train_pred = forward(train_data)[1]
        test_pred = forward(test_data)[1]
        train_accuracy = compute_accuracy(train_pred, train_target)
        test_accuracy = compute_accuracy(test_pred, test_target)

        print("After epoch {}: train acc {:.1f}%, test acc {:.1f}%".format(
            epoch + 1, 100 * train_accuracy, 100 * test_accuracy))

    return tuple(weights + biases), [100 * train_accuracy, 100 * test_accuracy]

def compute_accuracy(predictions, y_true):
    sum =  0
    n_samples = predictions.shape[0]
    for i in range(n_samples):
        if np.argmax(predictions[i]) == y_true[i]:
            sum += 1
    return sum / n_samples


def compute_gradients(X, hidden, pred, Y, weights):

    n_samples = X.shape[0]
    one_hot_Y = np.zeros_like(pred)
    one_hot_Y[np.arange(n_samples), Y] = 1

    error_out = pred - one_hot_Y

    W1_grad = (hidden.T @ error_out) / n_samples
    b1_grad = np.sum(error_out, axis=0) / n_samples

    error_hidden = error_out @ weights[1].T     #Backpropagation
    error_hidden[hidden <= 0] = 0   #ReLU

    W0_grad = (X.T @ error_hidden) / n_samples
    b0_grad = np.sum(error_hidden, axis=0) / n_samples

    return W0_grad, b0_grad, W1_grad, b1_grad





if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    parameters, metrics = main(main_args)
    print("Learned parameters:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in ws.ravel()[:12]] + ["..."]) for ws in parameters), sep="\n")
