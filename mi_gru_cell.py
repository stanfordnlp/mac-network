import tensorflow as tf
import numpy as np

class MiGRUCell(tf.compat.v1.nn.rnn_cell.RNNCell):
    def __init__(self, num_units, input_size = None, activation = tf.tanh, reuse = None):
        self.numUnits = num_units
        self.activation = activation
        self.reuse = reuse

    @property
    def state_size(self):
        return self.numUnits

    @property
    def output_size(self):
        return self.numUnits

    def mulWeights(self, inp, inDim, outDim, name = ""): 
        with tf.compat.v1.variable_scope("weights" + name):
            W = tf.compat.v1.get_variable("weights", shape = (inDim, outDim),
                initializer = tf.compat.v1.keras.initializers.glorot_normal())

        output = tf.matmul(inp, W)        
        return output

    def addBiases(self, inp1, inp2, dim, bInitial = 0, name = ""):
        with tf.compat.v1.variable_scope("additiveBiases" + name):
            b = tf.compat.v1.get_variable("biases", shape = (dim,), 
                initializer = tf.zeros_initializer()) + bInitial
        with tf.compat.v1.variable_scope("multiplicativeBias" + name):
            beta = tf.compat.v1.get_variable("biases", shape = (3 * dim,), 
                initializer = tf.ones_initializer())

        Wx, Uh, inter = tf.split(beta * tf.concat([inp1, inp2, inp1 * inp2], axis = 1), 
            num_or_size_splits = 3, axis = 1)
        output = Wx + Uh + inter + b        
        return output

    def __call__(self, inputs, state, scope = None):
        scope = scope or type(self).__name__
        with tf.compat.v1.variable_scope(scope, reuse = self.reuse):
            inputSize = int(inputs.shape[1])
            
            Wxr = self.mulWeights(inputs, inputSize, self.numUnits, name = "Wxr")
            Uhr = self.mulWeights(state, self.numUnits, self.numUnits, name = "Uhr")
            
            r = tf.nn.sigmoid(self.addBiases(Wxr, Uhr, self.numUnits, bInitial = 1, name = "r"))
            
            Wxu = self.mulWeights(inputs, inputSize, self.numUnits, name = "Wxu")
            Uhu = self.mulWeights(state, self.numUnits, self.numUnits, name = "Uhu")
            
            u = tf.nn.sigmoid(self.addBiases(Wxu, Uhu, self.numUnits, bInitial = 1, name = "u"))
            # r, u = tf.split(gates, num_or_size_splits = 2, axis = 1)

            Wx = self.mulWeights(inputs, inputSize, self.numUnits, name = "Wxl")
            Urh = self.mulWeights(r * state, self.numUnits, self.numUnits, name = "Uhl")
            c = self.activation(self.addBiases(Wx, Urh, self.numUnits, name = "2"))

            newH = u * state + (1 - u) * c # switch u and 1-u?
        return newH, newH

    def zero_state(self, batchSize, dtype = tf.float32):
        return tf.zeros((batchSize, self.numUnits), dtype = dtype)
        