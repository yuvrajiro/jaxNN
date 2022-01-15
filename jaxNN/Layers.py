# Layers
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

class layer:
    """
    A class for creating Layers

    ...

    Attributes
    ----------
    ninput : int
        Number of Input Nodes
    noutput : int
        Number of Output Nodes
    name : str
        Type of Layer

    Methods
    -------
    get_layer(name)
        Returns the Layer Object

    check_dimensions(x)
        Checks the Dimensions of the Input

    linear(x)
        Linear Layer

    relu(x)
        ReLU Layer

    sigmoid(x)
        Sigmoid Layer


    """
    def __init__(self, ninput, noutput, name="LINEAR"):

        """
        Parameters
        ----------
        ninput : int
            The number of input nodes
        noutput : int
            The number of output nodes
        name : str
            Name of the layer
            For Linear Layer, name = "LINEAR"
            For ReLU Layer, name = "RELU"
            For Sigmoid Layer, name = "SIGMOID"
            For Softmax Layer, name = "SOFTMAX"

        """


        self.ninput = ninput
        self.noutput = noutput
        self.current_layer = self.get_layer(name)

    def get_layer(self, name):
        """
        Returns the Layer Object
        :param name: Name of the Layer
        :return: Layer Object
        """
        return {
            "LINEAR": self.linear,
            "RELU": self.relu,
            "SIGMOID": self.sigmoid,
            "SOFTMAX": self.softmax
        }[name]

    def check_dimensions(self, x, w, b):
        """
        Checks the Dimensions of the Input
        :param x: Input Nodes
        :param w: weights
        :param b: bias for the layer
        :return: True if dimensions are equal else False

        """
        if x.shape[1] == w.shape[0]:
            return True
        else:
            return False

    def linear(self, input, weights, bias):

        """
        Parameters
        :param input: Input Nodes
        :param weights: weights
        :param bias: bias for the layer
        :return: A Linear Layer

        :raise Exception: If the dimensions of the input and weights are not equal

        """
        self.name = "linear"
        if self.check_dimensions(input, weights, bias):
            return jnp.dot(input, weights) + bias
        else:
            raise Exception(f"DIMENSION MISMATCH in layer.linear : input size is" +
                            f" {input.shape} and weight's dimension is {weights.shape} and bias {bias.shape}")

    def relu(self, input, weights, bias):

        """

        :param input: Input Nodes
        :param weights: weights
        :param bias: bias for the layer
        :return: A ReLU Layer

        :raise Exception: If the dimensions of the input and weights are not equal

        """
        self.name = "relu"
        if self.check_dimensions(input, weights, bias):
            return activation.relu(jnp.matmul(input, weights) + bias)
        else:
            raise Exception(f"DIMENSION MISMATCH in layer.relu : input size is" +
                            f" {input.shape} and weight's dimension is {weights.shape} and bias {bias.shape}")

    def sigmoid(self, input, weights, bias):
        """

        :param input: Input Nodes
        :param weights: weights
        :param bias: bias for the layer
        :return: A Sigmoid Layer

        :raise Exception: If the dimensions of the input and weights are not equal
        """
        self.name = "sigmoid"
        if self.check_dimensions(input, weights, bias):
            return activation.sigmoid(jnp.matmul(input, weights) + bias)
        else:
            raise Exception(f"DIMENSION MISMATCH in layer.sigmoid : input size is" +
                            f" {input.shape} and weight's dimension is {weights.shape} and bias {bias.shape}")

    def softmax(self, input, weights, bias):

        """

        :param input: Input Nodes
        :param weights: weights
        :param bias: bias for the layer
        :return: A Softmax Layer

        :raise Exception: If the dimensions of the input and weights are not equal
        """
        self.name = "softmax"
        if self.check_dimensions(input, weights, bias):
            return activation.softmax(jnp.matmul(input, weights) + bias)
        else:
            raise Exception(f"DIMENSION MISMATCH in layer.softmax : input size is" +
                            f" {input.shape} and weight's dimension is {weights.shape} and bias {bias.shape}")