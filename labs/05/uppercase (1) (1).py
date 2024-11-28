#!/usr/bin/env python3
# f5419161-0138-4909-8252-ba9794a63e53
# 4b50a6fb-a4a6-4b30-9879-0b671f941a72
import argparse
import datetime
import os
import re
os.environ.setdefault("KERAS_BACKEND", "torch")  # Use PyTorch backend unless specified otherwise

import keras
import torch
import numpy as np
from uppercase_data import UppercaseData

# TODO: Set reasonable values for the hyperparameters, especially for
# `alphabet_size`, `batch_size`, `epochs`, and `window`.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=85, type=int, help="If given, use this many most frequent chars.")
parser.add_argument("--batch_size", default=2000, type=int, help="Batch size.")#10000
parser.add_argument("--epochs", default=16, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay strength.")
parser.add_argument("--threads", default=16, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=25, type=int, help="Window size to use.")#100
parser.add_argument("--hidden_layers", default=1, nargs="*", type=int, help="Hidden layer sizes.")
parser.add_argument("--hidden_layer", default=460, nargs="*", type=int, help="Hidden layer sizes.")
parser.add_argument("--model", default="98,42.keras", type=str, help="Output model path.")
parser.add_argument("--save", default=False, type=bool, help="Save the model.")
parser.add_argument("--train", default=False, type=bool, help="Train the saved model")
parser.add_argument("--load", default="", type=str, help="Load")
parser.add_argument("--dropout", default=0.18, type=float, help="Dropout regularization.")
parser.add_argument("--label_smoothing", default=0.005, type=float, help="Label smoothing.")
parser.add_argument("--ensemble", default=False, type=bool, help="Ensemble model")
class TorchTensorBoardCallback(keras.callbacks.Callback):
    def __init__(self, path):
        self._path = path
        self._writers = {}

    def writer(self, writer): 
        if writer not in self._writers:
            import torch.utils.tensorboard
            self._writers[writer] = torch.utils.tensorboard.SummaryWriter(os.path.join(self._path, writer))
        return self._writers[writer]

    def add_logs(self, writer, logs, step):
        if logs:
            for key, value in logs.items():
                self.writer(writer).add_scalar(key, value, step)
            self.writer(writer).flush()

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            if isinstance(getattr(self.model, "optimizer", None), keras.optimizers.Optimizer):
                logs = logs | {"learning_rate": keras.ops.convert_to_numpy(self.model.optimizer.learning_rate)}
            self.add_logs("train", {k: v for k, v in logs.items() if not k.startswith("val_")}, epoch + 1)
            self.add_logs("val", {k[4:]: v for k, v in logs.items() if k.startswith("val_")}, epoch + 1)


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load data
    uppercase_data = UppercaseData(args.window, args.alphabet_size)

    # TODO: Implement a suitable model, optionally including regularization, select
    # good hyperparameters and train the model.
    #
    # The inputs are _windows_ of fixed size (`args.window` characters on the left,
    # the character in question, and `args.window` characters on the right), where
    # each character is represented by a "int32" index. To suitably represent
    # the characters, you can:
    # - Convert the character indices into _one-hot encoding_, which you can achieve by
    #   - suitable application of the `layers.CategoryEncoding` layer
    #   - when using Functional API, any `keras.ops` can be used as a Keras layer:
    #       inputs = keras.layers.Input(shape=[2 * args.window + 1], dtype="int32")
    #       encoded = keras.ops.one_hot(inputs, len(uppercase_data.train.alphabet))
    #   You can then flatten the one-hot encoded windows and follow with a dense layer.
    # - Alternatively, you can use `keras.layers.Embedding` (which is an efficient
    #   implementation of one-hot encoding followed by a Dense layer) and flatten afterwards.
    if args.load =="" and not args.ensemble:
        inputs = keras.Input(shape=[2 * args.window + 1],dtype="int32")
        encoded = keras.ops.one_hot(inputs,len(uppercase_data.train.alphabet))
        hidden = keras.layers.Flatten()(encoded)
        hidden = keras.layers.Dropout(args.dropout)(hidden)
        for hidden_layer in range(args.hidden_layers):
            hidden = keras.layers.Dense(args.hidden_layer,activation="relu")(hidden)
            hidden = keras.layers.Dropout(args.dropout)(hidden)
        outputs = keras.layers.Dense(2,activation="softmax")(hidden)
        model = keras.Model(inputs=inputs,outputs=outputs)
        
        uppercase_data.dev.data["labels"] = keras.utils.to_categorical(uppercase_data.dev.data["labels"])
        uppercase_data.train.data["labels"] = keras.utils.to_categorical(uppercase_data.train.data["labels"])
        uppercase_data.test.data["labels"]= keras.utils.to_categorical(uppercase_data.test.data["labels"])
        model.summary()
        opti = keras.optimizers.AdamW(weight_decay=args.weight_decay)#Adam
        opti.exclude_from_weight_decay(var_names=["bias"])
        #opti = keras.optimizers.Adam()
        model.compile(
            optimizer=opti,
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),#categorical 
            metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")]#categorical
        )
        model.fit(
            uppercase_data.train.data["windows"], uppercase_data.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs, shuffle=False,
            validation_data=(uppercase_data.dev.data["windows"],uppercase_data.dev.data["labels"]), 
        )
        #model.evaluate(
        #    uppercase_data.test.data["windows"], uppercase_data.test.data["labels"], batch_size=args.batch_size,
        #)
        if(args.save):
            model.save(args.model)
    elif args.load != "" and not args.ensemble :
        model = keras.models.load_model(args.load,compile=False)
        model.summary()
        if args.train:
            #opti = keras.optimizers.Adam()
            opti = keras.optimizers.AdamW(weight_decay=args.weight_decay)#Adam
            opti.exclude_from_weight_decay(var_names=["bias"])
            uppercase_data.dev.data["labels"] = keras.utils.to_categorical(uppercase_data.dev.data["labels"])
            uppercase_data.train.data["labels"] = keras.utils.to_categorical(uppercase_data.train.data["labels"])
            uppercase_data.test.data["labels"]= keras.utils.to_categorical(uppercase_data.test.data["labels"])
            model.compile(
                optimizer=opti,
                loss=keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),#categorical 
                metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")]#categorical
            )
            model.fit(
                uppercase_data.train.data["windows"], uppercase_data.train.data["labels"],
                batch_size=args.batch_size, epochs=args.epochs,
                validation_data=(uppercase_data.dev.data["windows"],uppercase_data.dev.data["labels"]), 
            )
            if(args.save):
                model.save(args.model)

    models = []
    if args.ensemble:
        models.append(keras.models.load_model("98,44.keras",compile=False))
        models.append(keras.models.load_model("98,42.keras",compile=False))
        models.append(keras.models.load_model("2.keras",compile=False))
        """for i in models:
            i.summary()
        input = keras.Input(shape=[2 * args.window + 1],dtype="int32")
        #hidden = [m(input) for m in models]
        m1 = models[0](input)
        m2 = models[1](input)
        m3 = models[2](input)
        output = keras.layers.Average()([m1, m2, m3])
        model = keras.Model(inputs=input, outputs=output)
        model.compile(
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing), 
            metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")]
        )"""
        predictions =  [np.argmax(m.predict(uppercase_data.test.data["windows"]),axis=1) for m in models]
        os.makedirs(args.logdir, exist_ok=True)
        with open(os.path.join(args.logdir, "uppercase_test.txt"), "w", encoding="utf-8") as predictions_file:
            for letter,s1,s2,s3 in zip(uppercase_data.test.text,predictions[0],predictions[1],predictions[2]):
                if s1+s2+s3 >= 2:
                    predictions_file.write(letter.upper())
                else:
                    predictions_file.write(letter.lower())
        return
    """
    # TODO: Generate correctly capitalized test set.
    # Use `uppercase_data.test.text` as input, capitalize suitable characters,
    # and write the result to predictions_file (which is
    # `uppercase_test.txt` in the `args.logdir` directory).
    os.makedirs(args.logdir, exist_ok=True)
    #uppercase_data.evaluate(uppercase_data.test,model.predict(uppercase_data.test.data["windows"]))
    predicted_test= np.argmax(model.predict(uppercase_data.test.data["windows"]),axis=1)
    #predicted_test = torch.argmax()
    with open(os.path.join(args.logdir, "uppercase_test.txt"), "w", encoding="utf-8") as predictions_file:
        for letter,size in zip(uppercase_data.test.text,predicted_test):
            if size == 1:
                predictions_file.write(letter.upper())
            else:
                predictions_file.write(letter.lower())
    """


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
