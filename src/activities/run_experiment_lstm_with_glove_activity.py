import os
from dataclasses import dataclass
from temporalio import activity

import numpy as np
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
tf.config.run_functions_eagerly(True)
tf.compat.v1.enable_eager_execution()

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
  Input,
  Embedding,
  LSTM,
  Dense,
  Dropout,
  GlobalAveragePooling1D,
)

from src.utils.calculate_metrics import calculate_metrics, EvaluationData
from src.utils.convert_to_native import convert_to_native

@dataclass
class RunExperimentLSTMWithGloveIn:
  x_seq_path: str
  y_path: str
  embedding_matrix_path: str
  max_len: int
  num_words: int
  embedding_dim: int
  lstm_units: int
  lstm_dropout: float
  lstm_recurrent_dropout: float
  pool_dropout: float
  dense_units: int
  dense_activation: str
  batch_size: int
  epochs: int
  learning_rate: float
  loss: str
  metrics: list
  n_splits: int
  random_state: int
  verbose: int
  class_weight_0: float
  class_weight_1: float

@dataclass
class RunExperimentLSTMWithGloveOut:
  metrics: EvaluationData

def _build_model(
  max_len: int,
  num_words: int,
  embedding_dim: int,
  embedding_matrix: np.ndarray,
  lstm_units: int,
  lstm_dropout: float,
  lstm_recurrent_dropout: float,
  pool_dropout: float,
  dense_units: int,
  dense_activation: str,
):
  inputs = Input(shape=(max_len,))
  embedding = Embedding(
    input_dim=num_words,
    output_dim=embedding_dim,
    weights=[embedding_matrix],
    input_length=max_len,
    trainable=False
  )(inputs)

  x = LSTM(
    lstm_units,
    dropout=lstm_dropout,
    recurrent_dropout=lstm_recurrent_dropout,
    return_sequences=True
  )(embedding)
  x = GlobalAveragePooling1D()(x)
  x = Dropout(pool_dropout)(x)
  outputs = Dense(dense_units, activation=dense_activation)(x)
  return Model(inputs, outputs)

@activity.defn
async def run_experiment_lstm_with_glove_activity(data: RunExperimentLSTMWithGloveIn) -> RunExperimentLSTMWithGloveOut:
  x_seq = np.load(data.x_seq_path)
  y = np.load(data.y_path)
  embedding_matrix = np.load(data.embedding_matrix_path)

  optimizer = Adam(learning_rate=data.learning_rate)

  model = _build_model(
    max_len=data.max_len,
    num_words=data.num_words,
    embedding_dim=data.embedding_dim,
    embedding_matrix=embedding_matrix,
    lstm_units=data.lstm_units,
    lstm_dropout=data.lstm_dropout,
    lstm_recurrent_dropout=data.lstm_recurrent_dropout,
    pool_dropout=data.pool_dropout,
    dense_units=data.dense_units,
    dense_activation=data.dense_activation,
  )
  model.compile(
      optimizer=optimizer,
      loss=data.loss,
      metrics=data.metrics
  )

  skf = StratifiedKFold(n_splits=data.n_splits, shuffle=True, random_state=data.random_state)
  y_scores = np.zeros_like(y, dtype=float)

  for fold, (train_idx, val_idx) in enumerate(skf.split(x_seq, y)):
    print(f"Fold {fold+1}/{data.n_splits}")
    x_tr, x_val_k = x_seq[train_idx], x_seq[val_idx]
    y_tr, y_val_k = y[train_idx], y[val_idx]

    model_fold = _build_model(
      max_len=data.max_len,
      num_words=data.num_words,
      embedding_dim=data.embedding_dim,
      embedding_matrix=embedding_matrix,
      lstm_units=data.lstm_units,
      lstm_dropout=data.lstm_dropout,
      lstm_recurrent_dropout=data.lstm_recurrent_dropout,
      pool_dropout=data.pool_dropout,
      dense_units=data.dense_units,
      dense_activation=data.dense_activation,
    )
    optimizer_fold = Adam(learning_rate=data.learning_rate)
    model_fold.compile(
      optimizer=optimizer_fold,
      loss=data.loss,
      metrics=data.metrics
    )
    model_fold.fit(
      x_tr, y_tr,
      validation_data=(x_val_k, y_val_k),
      batch_size=data.batch_size,
      epochs=data.epochs,
      verbose=data.verbose,
      class_weight={0: data.class_weight_0, 1: data.class_weight_1}
    )
    y_scores[val_idx] = model_fold.predict(x_val_k).flatten()

  metrics = calculate_metrics(y, y_scores)

  return RunExperimentLSTMWithGloveOut(metrics=convert_to_native(metrics))
