# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 13:20:17 2020

@author: loram
"""
import tensorflow as tf
import tensorflow.keras as keras

class CategoricalActivation(keras.layers.Layer):
    """
    A categorical activation layer class. Inherited the Keras Layer class.
    This takes a dense layer, of size n, where n is the number of categories in the variable
     and outputs a near to discrete sample from that variable.
    """
    def __init__(self):
        """
        Initiate activation layer.
        """
        super(CategoricalActivation, self).__init__()

    def call(self, input, temperature, eps=1e-20):"""
        Applies the Gumbel-softmax trick to the output of a dense layer.
        This is a continuous approximation for sampling from the categorical distribution,
         where the dense layer consists of the logits for the distribution.
            
        Parameters
        ----------
        input : the output of a dense layer
        temperature : the temperature used, this determines how close to discrete the output is
        eps : size of epsilon in the gumbel sampling
        
        Returns
        -------
        A vector that is very close to a one-hot encoded vector, that represents a category.
        """
        U = tf.random.uniform(tf.shape(input), minval=0, maxval=1)
        y = input + -tf.math.log(-tf.math.log(U + eps) + eps)
        activation = tf.keras.layers.Softmax()
        output = activation(y / temperature)
        return output


class MultiCategorical(keras.layers.Layer):
    """
    A multi-categorical Layer class. Inherits the Keras Layer class.
    For a data sample with k variables, it passes the input into k parallel dense layers.
    Each dense layer is the size of the number of categories in the corresponding variable.
    The dense layers are each passed through a categorical activation function.
    The output is a multi-categorical data sample.
    """
    def __init__(self, temperature, variable_sizes):
        """
        Initialise layer.
        Parameters
        ----------
        temperature : sets the temperature used in the categorical activation layer.
        variable_sizes : list of the sizes of each categorical variable to output
        """
        super(MultiCategorical, self).__init__()

        self.dense_layers = []
        self.activation_layers = []
        self.temperature = temperature

        for i, size in enumerate(variable_sizes):
            dense = keras.layers.Dense(size)
            self.dense_layers.append(dense)

        for i, size in enumerate(variable_sizes):
            activation = CategoricalActivation()
            self.activation_layers.append(activation)

    def call(self, inputs, training=False):
        """
        Parameters
        ----------
        inputs : the output from a dense layer
        training : True if training

        Returns
        -------
        output: a multi-categorical sample
        """
        outputs = []
        for layer, activation in zip(self.dense_layers, self.activation_layers):
            logits = layer(inputs)
            output = activation(logits, self.temperature)
            outputs.append(output)
        output = tf.concat(outputs, axis=1)
        return output

class Generator(keras.Model):
    """
    Generator model class. Inherits the Keras model class.
    The model consists of n hidden layers, followed by a MultiCategorical layer.
    The final output is a fake data sample.
    """
    def __init__(self, hidden_sizes, temperature, output_variable_sizes, conditional = False):
        """
        Initiate the generator model.

        Parameters
        ----------
        hidden_sizes : list of the sizes of hidden dense layers (implicitly sets the number of layers, n)
        temperature : the temperature used in the Gumbel-Softmax activation in the MultiCategorical layer
        output_variable_sizes : list of sizes of each categorical data variable in the generated data sample
        conditional : True if the generator is conditional, taking in a conditional label y during training
        """
        super().__init__(name="generator")
        
        self.conditional = conditional
        self.hidden_sizes = hidden_sizes
        self.dense_layer_list = []
        self.norm_layer_list = []
        self.relu_layer_list = []
        
        for i, size in enumerate(hidden_sizes):
            self.dense_layer_list.append(keras.layers.Dense(hidden_sizes[i]))
            self.norm_layer_list.append(keras.layers.BatchNormalization())
            self.relu_layer_list.append(keras.layers.ReLU())

        self.output_layer = MultiCategorical(temperature, output_variable_sizes)
       
    def call(self, inputs, labels = None, training=False):
        """
        Parameters
        ----------
        inputs : a vector of random noise from specified distribution
        labels : if the generator is conditional, this is the variable that it is conditioned on
        training : True if model is training, used for batch normalisation

        Returns
        -------
        output: the final output from the generator, a fake data sample
        """

        if self.conditional == True:
            x = tf.concat([inputs, labels], axis = 1)
        else:
            x = inputs
        
        for i, size in enumerate(self.hidden_sizes):
            x = self.dense_layer_list[i](x)
            x = self.norm_layer_list[i](x, training = training)
            x = self.relu_layer_list[i](x)
        
        output = self.output_layer(x)
        
        if self.conditional == True:
            output = tf.concat([output, labels], axis = 1)
            
        return output
    
class Discriminator(keras.Model):
    """
    Discriminator model class. Inherits the Keras model class.
    The model consists of n hidden layers, followed by a dense layer of size one.
    The final output is a probability that the input is a real sample.
    """
    def __init__(self, hidden_sizes):
        """
        Initiate the discriminator model.

        Parameters
        ----------
        hidden_sizes : list of the sizes of hidden dense layers (implicitly sets the number of layers, n)
        """
        super().__init__(name="discriminator")

        self.hidden_sizes = hidden_sizes
        self.dense_layer_list = []
        self.norm_layer_list = []
        self.relu_layer_list = []
        
        for i, size in enumerate(hidden_sizes):
            self.dense_layer_list.append(keras.layers.Dense(hidden_sizes[i]))
            self.norm_layer_list.append(keras.layers.BatchNormalization())
            self.relu_layer_list.append(keras.layers.ReLU())

        self.dense_final = keras.layers.Dense(1)
        
    def call(self, inputs, training=False):
        """
        Parameters
        ----------
        inputs : an array of random noise from a specified distribution
        training : True if model is training, used for batch normalisation

        Returns
        -------
        x: the output of the discriminator model, a fake data sample
        """
        x = inputs
        for i, size in enumerate(self.hidden_sizes):
            x = self.dense_layer_list[i](x)
            x = self.norm_layer_list[i](x, training = training)
            x = self.relu_layer_list[i](x)
        x = self.dense_final(x)
        return x