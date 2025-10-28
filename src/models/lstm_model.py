import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
)

from .attention import BahdanauAttention


def build_model(
    embedding_matrix: np.ndarray,
    num_words: int,
    embedding_dim: int,
    max_len: int,
    bidirectional: bool = True,
    use_attention: bool = True
) -> Model:
    inputs = Input(shape=(max_len,))
    embedding = Embedding(
        input_dim=num_words, 
        output_dim=embedding_dim, 
        weights=[embedding_matrix], 
        input_length=max_len, 
        trainable=False
    )(inputs)

    if bidirectional:
        x = Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(embedding)
    else:
        x = LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(embedding)

    if use_attention:
        attention = BahdanauAttention(100)
        x = attention(x, x)
    else:
        x = GlobalAveragePooling1D()(x)

    x = Dropout(0.5 if not bidirectional else 0.02)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    return model
