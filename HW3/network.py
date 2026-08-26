"""
network.py (Fall 2026)

Neural network code from Mike Nielsen's NNDL book (Chapter 1, "network.py").
Modified slightly for our course.
"""

import random
import json
import numpy as np
import pandas as pd


class Network(object):
    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network, for example [2, 3, 1].
        The biases and weights are initialized randomly.  Note that
        the first layer is an input layer, and by convention we
        won't set any biases for those neurons."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        ##
        self.init_acts_shape = []

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)``.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch -- essentially the role as a validation data."""
        n = len(training_data)
        if test_data:
            nvalid = len(test_data)

        for j in range(epochs):
            # random.shuffle(training_data) # supressed for now
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            # Compute the training result for this epoch.
            acc_train = self.evaluate(training_data)/n
            if not test_data:
                print("Epoch {}: train acc {:.4f}".format(j, acc_train))
            else:
                acc_valid = self.evaluate(test_data)/nvalid
                print("Epoch {}: train acc {:.4f}, valid acc {:.4f}".format
                      (j, acc_train, acc_valid))
            # Early exit if applies
            if acc_train == 1.0:
                break

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [
            w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # forward pass
        activation = x
        activations = [x]  # list to store all the activations, layer by layer

        ## nt: DO NOT REMOVE THIS LINE!!
        self.init_acts_shape = [act.shape for act in activations]

        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result."""
        test_results = [
            (np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data
        ]
        return sum(int(x == y) for (x, y) in test_results)

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input.  Note this
        function is called during evaluation; not during training/backprop."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \\partial C_x /
        \\partial a for the output activations."""
        return output_activations - y

    @classmethod
    def load_network(cls, filename):
        """Load a neural network from a json file ``filename``.  Returns an
        instance of Network. """
        f = open(filename, "r")
        data = json.load(f)
        f.close()
        net = cls(data["sizes"])
        net.weights = [np.array(w) for w in data["weights"]]
        net.biases = [np.array(b) for b in data["biases"]]
        return net

    def save_network(self, filename):
        """Save the neural network to a json file ``filename``."""
        data = {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases]  # ,
            # "cost": str(self.cost.__name__)
        }
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))

def vectorize_target(n, target):
    """Return an array of shape (n,1) with a 1.0 in the target position
    and zeroes elsewhere.  The parameter target is assumed to be
    an array of size 1, and the 0th item is the target position (1). """
    e = np.zeros((n, 1))
    e[int(target[0])] = 1.0
    return e


def my_load_csv(fname, input_size, target_size, seednum=17):
    ''' Function to load the data from a csv file.  Note the target (y)
        is assumed to be already in the one-hot-vector notation.
        Also each instance in the returned data is made into column vectors.'''
    # Read in the data into pandas dataframe
    df = pd.read_csv(fname, header=None)

    # Set the random seed if specified to shuffle, for reproducibility.
    # Otherwise no shuffling.
    if seednum:
        df = df.sample(frac=1, random_state=seednum)

    # Separate the X and Y parts
    X = df[df.columns[:input_size]].values.tolist()
    Y = df[df.columns[-target_size:]].values.tolist()

    # Combine the parts for each instance and put all in a list.
    # Note: x and y are both converted into a column vector/array.
    dataset = [(np.reshape(x, (input_size, 1)), np.reshape(y, (target_size, 1)))
               for x, y in zip(X, Y)]
    return dataset
