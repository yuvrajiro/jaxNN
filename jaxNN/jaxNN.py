import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

class jaxNN():

    """
    A simple neural network class using JAX from scratch that can be used to
    train, predict and evaluate a neural network.

    Parameters
    ----------
    layers : list
        A list of the layers of the network.
    random_key : int
        A random key to be used for the random number generator.

    Attributes
    ----------
    list_of_layers : list
        A list of the layers of the network.
    nlayers : int
        The number of layers in the network.
    key : int
        A random key to be used for the random number generator.
    weights : list
        A list of the weights of the network.
    biases : list
        A list of the biases of the network.

    Methods
    -------
    forward_prop(x)
        A method to forward propagate the input through the network.

    backward_prop(x, y)
        A method to back propagate the input through the network.

    predict(x)
        A method to predict the output of the network.

    evaluate(x, y)
        A method to evaluate the network.

    train(x, y, epochs, batch_size, learning_rate, optimizer, loss_function)
        A method to train the network.

    """



    def __init__(self, list_of_layers, random_key=42):
        """
        A simple neural network class using JAX from scratch
        :param list_of_layers: A list of the layers of the network
        :param random_key: A random key to be used for the random number
        generator

        """
        self.list_of_layers = list_of_layers
        self.nlayers = len(list_of_layers)
        self.key = random.PRNGKey(random_key)
        self.weights, self.bias = self.init_weights()
        self.check_input_output_dimension()

    def check_input_output_dimension(self):
        """
        A method to check the input and output dimension of the network.
        :return: None
        :raise: Exception
        """
        list_of_layers = self.list_of_layers
        for idx, layer in enumerate(list_of_layers):
            if idx == 0:
                pass
            else:
                if layer.ninput != list_of_layers[idx - 1].noutput:
                    raise Exception(f"Input of {idx} Layer is not equal to output of prev")

    def init_weights(self):
        """
        A method to initialize the weights and biases of the network.
        :return: A list of weights and a list of biases
        """
        key = self.key
        Weights = []
        bias = []
        for layer in self.list_of_layers:
            Weights.append(random.normal(key, (layer.ninput, layer.noutput)))
            bias.append(random.normal(key, (layer.noutput,)))
        return Weights, bias

    def forward_pass(self, x, weights, biass):
        """
        A method to forward propagate the input through the network.
        :param x: The input to the network
        :param weights: The weights of the network
        :param biass: The biases of the network
        :return: The output of the network
        """
        list_of_layers = self.list_of_layers
        for weight, bias, layer in zip(weights, biass, list_of_layers):
            x = layer.current_layer(x, weight, bias)
        return x

    def train(self, x, y, epoch = 100, batch_size = 100, learning_rate = 0.01, optimizer = "SGD", loss_function = "MSE"):
        """
        A method to train the network.
        :param x:
        :param y:
        :param epoch:
        :param batch_size:
        :param learning_rate:
        :param optimizer:
        :param loss_function:
        :return:
        """
        self.weights, self.bias = self.backward_pass(x, y)
        self.x = x
        self.y = y

        for epoch in range(epoch):
            W_grad, b_grad = grad(self.loss, (0, 1))(self.weights, self.bias, x, y)
            print(self.weights)
            idx = 0
            for w, b in zip(W_grad, b_grad):
                self.weights[idx] = self.weights[idx] - 0.01 * w
                self.bias[idx] = self.bias[idx] - 0.0001 * b
                idx += 1
        return self

    def loss(self, W, b, x, y):
        """
        A method to calculate the loss of the network.
        :param W: The weights of the network
        :param b: The biases of the network
        :param x: The input to the network
        :param y: The output of the network
        :return:  The loss of the network
        """
        y_pred = self.forward_pass(x, W, b)
        error = ((y_pred.flatten() - y))
        return jnp.mean(jnp.square(error))



