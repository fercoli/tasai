import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import and work on the data


#the intellect
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

##Builder for a general forward ANN
#n_dense        :   number of layers                                                {int}
#n_nodes        :   the number of nodes to use for each layer                       {list/array of ints}
#dropouts_rate  :   the rate of dropout for the given layer                         {list/array of floats}
#activations    :   the activation function to use                                  {list/array of strings}
#in_shape       :   the input shape, this should have at least the number of inputs {tuple/array/tensor of ints[probably]}
#kernel_int     :   the initial values for the kernel matrix                        {list/array of strings}
#opti           :   optimizer to use, for default we use adam                       {string}
#ls             :   the loss function to use in the backpropagation                 {string}
#met            :   metrics to optimize, for default we use ["accuracy"]            {list/array of string}
def Builder(n_dense, n_nodes, activations, in_shape,opti = "adam", ls = "binary_crossentropy", met = ["accuracy"], dropouts_rate = None, kernel_int = None):
    pricer = Sequential()

    #for default the initializer is uniform
    if not kernel_int:
        kernel_int = []
        for _ in range(n_dense):
            kernel_int.append("uniform")

    #for default the dropout rate is 0
    if not dropouts_rate:
        dropouts_rate = []
        for _ in range(n_dense):
            dropouts_rate.append(0)

    for i in range(n_dense):
        if i == 0:
            pricer.add(Dense(units = n_nodes[i], activation = activations[i], kernel_initializer = kernel_int[i], input_dim = in_shape))
        else:
            pricer.add(Dense(units = n_nodes[i], activation = activations[i], kernel_initializer = kernel_int[i]))
            pricer.add(Dropout(rate = dropouts_rate[i]))
    pricer.compile(optimizer = opti, loss = ls, metrics = met)
    return pricer

if __name__ == "__main__":
    Bravo = Builder(3,[3,3,3],["relu","relu","sigmoid"],(5,30))
