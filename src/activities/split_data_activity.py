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

  # Check if stratified splitting is possible
  # Need at least 2 samples per class for stratification
  unique_classes, class_counts = np.unique(y, return_counts=True)
  min_class_count = np.min(class_counts)
  
  # Use stratified split if possible, otherwise use random split
  use_stratify = min_class_count >= 2
  
  if use_stratify:
    print(f"Using stratified split (min class count: {min_class_count})")
    x_train_full, x_temp, y_train_full, y_temp = train_test_split(
      x_seq, y, test_size=0.3, stratify=y, random_state=data.random_state
    )
    
    # Check if second split can also be stratified
    unique_classes_temp, class_counts_temp = np.unique(y_temp, return_counts=True)
    min_class_count_temp = np.min(class_counts_temp)
    
    if min_class_count_temp >= 2:
      x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=(1/3), stratify=y_temp, random_state=data.random_state
      )
    else:
      print(f"Using random split for validation/test (min class count: {min_class_count_temp})")
      x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=(1/3), random_state=data.random_state
      )
  else:
    print(f"Using random split (min class count: {min_class_count})")
    x_train_full, x_temp, y_train_full, y_temp = train_test_split(
      x_seq, y, test_size=0.3, random_state=data.random_state
    )
    
    x_val, x_test, y_val, y_test = train_test_split(
      x_temp, y_temp, test_size=(1/3), random_state=data.random_state
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
