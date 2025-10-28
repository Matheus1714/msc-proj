import numpy as np
from temporalio import activity
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from constants import (
  X_TRAIN_PATH,
  X_VAL_PATH,
  X_TEST_PATH,
  Y_TRAIN_PATH,
  Y_VAL_PATH,
  Y_TEST_PATH,
)

@dataclass
class SplitDataIn:
  x_seq_path: str
  y_path: str
  random_state: int

@dataclass
class SplitDataOut:
  x_train_path: str
  x_val_path: str
  x_test_path: str
  y_train_path: str
  y_val_path: str
  y_test_path: str

@activity.defn
async def split_data_activity(data: SplitDataIn) -> SplitDataOut:
  x_seq = np.load(data.x_seq_path)
  y = np.load(data.y_path)

  x_train_full, x_temp, y_train_full, y_temp = train_test_split(
    x_seq, y, test_size=0.3, stratify=y, random_state=data.random_state
  )

  x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=(1/3), stratify=y_temp, random_state=data.random_state
  )

  np.save(X_TRAIN_PATH, x_train_full)
  np.save(X_VAL_PATH,   x_val)
  np.save(X_TEST_PATH,  x_test)
  np.save(Y_TRAIN_PATH, y_train_full)
  np.save(Y_VAL_PATH,   y_val)
  np.save(Y_TEST_PATH,  y_test)

  return SplitDataOut(
    x_train_path=X_TRAIN_PATH,
    x_val_path=X_VAL_PATH,
    x_test_path=X_TEST_PATH,
    y_train_path=Y_TRAIN_PATH,
    y_val_path=Y_VAL_PATH,
    y_test_path=Y_TEST_PATH,
  )
