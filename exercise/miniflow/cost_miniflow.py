"""
Finish the MSE method for calculating the cost
with the mean squared error.

Your code will go at the bottom of this file.
"""


import numpy as np

class Layer:
    def __init__(self, inbound_layers=[]):
        self.inbound_layers = inbound_layers
        self.value = None
        self.outbound_layers = []
        # New property! Keys are the inputs to this layer and
        # their values are the partials of this layer with
        # respect to that input.
        self.gradients = {}
        for layer in inbound_layers:
            layer.outbound_layers.append(self)

    def forward():
        raise NotImplementedError


class Input(Layer):
    def __init__(self):
        Layer.__init__(self)

    def forward(self):
        # Do nothing because nothing is calculated.
        pass


class Linear(Layer):
    def __init__(self, inbound_layer, weights, bias):
        Layer.__init__(self, [inbound_layer, weights, bias])

    def forward(self):
        inputs = self.inbound_layers[0].value
        weights = self.inbound_layers[1].value
        bias = self.inbound_layers[2].value
        self.value = np.dot(inputs, weights) + bias


class Sigmoid(Layer):
    def __init__(self, layer):
        Layer.__init__(self, [layer])

    def _sigmoid(self, x):
        """
        This method is separate from `forward` because it
        will be used with `backward` as well.

        `x`: A numpy array-like object.
        """
        return 1. / (1. + np.exp(-x))

    def forward(self):
        input_value = self.inbound_layers[0].value
        self.value = self._sigmoid(input_value)


def topological_sort(feed_dict):
    """
    Sort the layers in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` Layer and the value is the respective value feed to that Layer.

    Returns a list of sorted layers.
    """

    input_layers = [n for n in feed_dict.keys()]

    G = {}
    layers = [n for n in input_layers]
    while len(layers) > 0:
        n = layers.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_layers:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            layers.append(m)

    L = []
    S = set(input_layers)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_layers:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(output_layer, sorted_layers):
    """
    Performs a forward pass through a list of sorted Layers.

    Arguments:

        `output_layer`: A Layer in the graph, should be the output layer (have no outgoing edges).
        `sorted_layers`: a topologically sorted list of layers.

    Returns the output layer's value
    """

    for n in sorted_layers:
        n.forward()

    return output_layer.value


def MSE(computed_output, ideal_output, n_inputs):
    """
    Calculates the mean squared error.
    
    `computed_output`: a numpy array
    `ideal_output`: a numpy array
    `n_inputs`: the number of training inputs (not including weights
                or biases) to the network
    
    Return the mean squared error of output layer.
    
    You will want to use np.linalg.norm to make your calculation easier.
    
    Your code here!
    """
    res = np.linalg.norm(ideal_output - computed_output, axis = 1) ** 2
#     n = ideal_output.shape[0]
    res = res.sum()/(2.0 * n_inputs)
    return res