import pandas as pd
from typing import Tuple
from src.utils.integer_proportion_1_n import integer_proportion_1_n

def calculate_class_weights_from_csv(csv_path: str) -> Tuple[float, float]:
  df = pd.read_csv(csv_path)
  
  included = df["included"].astype(bool)

  count_class_0 = (~included).sum()
  count_class_1 = included.sum()
  
  if count_class_0 == 0 or count_class_1 == 0:
    return (1.0, 1.0)

  _, weight_1 = integer_proportion_1_n(count_class_1, count_class_0)
  return (1.0, float(weight_1))
