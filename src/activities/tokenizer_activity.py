import json
import pandas as pd
import numpy as np
import os
from temporalio import activity
from dataclasses import dataclass

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

@dataclass
class TokenizerIn:
  input_data_path: str
  tokenized_data_path: str
  word_index_path: str
  x_seq_path: str
  y_path: str
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
  os.makedirs(os.path.dirname(data.tokenized_data_path), exist_ok=True)
  os.makedirs(os.path.dirname(data.word_index_path), exist_ok=True)
  os.makedirs(os.path.dirname(data.x_seq_path), exist_ok=True)
  os.makedirs(os.path.dirname(data.y_path), exist_ok=True)
  
  df = pd.read_csv(data.input_data_path)

  y = np.array(df["included"].to_list())
  np.save(data.y_path, y)

  tokenizer = Tokenizer(num_words=data.max_words)
  tokenizer.fit_on_texts(df["text"].to_list())
  sequences = tokenizer.texts_to_sequences(df["text"].to_list())
  x_seq = pad_sequences(sequences, maxlen=data.max_len)

  df_tokenized = df.copy()
  df_tokenized["x_seq"] = list(x_seq)
  df_tokenized.to_csv(data.tokenized_data_path, index=False)

  np.save(data.x_seq_path, x_seq)

  with open(data.word_index_path, "w") as f:
      json.dump(tokenizer.word_index, f)

  return TokenizerOut(
      tokenized_data_path=data.tokenized_data_path,
      word_index_path=data.word_index_path,
      x_seq_path=data.x_seq_path,
      y_path=data.y_path,
  )
