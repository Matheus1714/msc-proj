import json
import pandas as pd
import numpy as np
from temporalio import activity
from dataclasses import dataclass

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from constants import (
  TOKENIZED_DATA_PATH,
  WORD_INDEX_PATH,
  X_SEQ_PATH,
  Y_PATH,
)

@dataclass
class TokenizerIn:
  input_data_path: str
  max_words: int
  max_len: int

@dataclass
class TokenizerOut:
  tokenized_data_path: str
  word_index_path: str
  x_seq_path: str
  y_path: str

@activity.defn
async def tokenizer_activity(data: TokenizerIn) -> TokenizerOut:
  df = pd.read_csv(data.input_data_path)

  y = np.array(df["included"].to_list())
  np.save(Y_PATH, y)

  tokenizer = Tokenizer(num_words=data.max_words)
  tokenizer.fit_on_texts(df["text"].to_list())
  sequences = tokenizer.texts_to_sequences(df["text"].to_list())
  x_seq = pad_sequences(sequences, maxlen=data.max_len)

  df_tokenized = df.copy()
  df_tokenized["x_seq"] = list(x_seq)
  df_tokenized.to_csv(TOKENIZED_DATA_PATH, index=False)

  np.save(X_SEQ_PATH, x_seq)

  with open(WORD_INDEX_PATH, "w") as f:
      json.dump(tokenizer.word_index, f)

  return TokenizerOut(
      tokenized_data_path=TOKENIZED_DATA_PATH,
      word_index_path=WORD_INDEX_PATH,
      x_seq_path=X_SEQ_PATH,
      y_path=Y_PATH,
  )
