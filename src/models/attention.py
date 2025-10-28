import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Softmax


class BahdanauAttention(Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)
        self.softmax = Softmax(axis=1)

    def call(self, query, values):
        score = tf.nn.tanh(self.W1(query) + self.W2(values))
        attention_weights = self.softmax(self.V(score))
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector
