import numpy as np
import os
from temporalio import activity
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class SplitDataIn:
  x_seq_path: str
  y_path: str
  x_train_path: str
  x_val_path: str
  x_test_path: str
  y_train_path: str
  y_val_path: str
  y_test_path: str
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
  os.makedirs(os.path.dirname(data.x_train_path), exist_ok=True)
  os.makedirs(os.path.dirname(data.x_val_path), exist_ok=True)
  os.makedirs(os.path.dirname(data.x_test_path), exist_ok=True)
  os.makedirs(os.path.dirname(data.y_train_path), exist_ok=True)
  os.makedirs(os.path.dirname(data.y_val_path), exist_ok=True)
  os.makedirs(os.path.dirname(data.y_test_path), exist_ok=True)
  
  x_seq = np.load(data.x_seq_path)
  y = np.load(data.y_path)

  x_train_full, x_temp, y_train_full, y_temp = train_test_split(
    x_seq, y, test_size=0.3, stratify=y, random_state=data.random_state
  )

  x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=(1/3), stratify=y_temp, random_state=data.random_state
  )

  np.save(data.x_train_path, x_train_full)
  np.save(data.x_val_path,   x_val)
  np.save(data.x_test_path,  x_test)
  np.save(data.y_train_path, y_train_full)
  np.save(data.y_val_path,   y_val)
  np.save(data.y_test_path,  y_test)

  return SplitDataOut(
    x_train_path=data.x_train_path,
    x_val_path=data.x_val_path,
    x_test_path=data.x_test_path,
    y_train_path=data.y_train_path,
    y_val_path=data.y_val_path,
    y_test_path=data.y_test_path,
  )
