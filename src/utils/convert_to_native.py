import numpy as np

def convert_to_native(obj):
  if isinstance(obj, dict):
    return {k: convert_to_native(v) for k, v in obj.items()}
  elif isinstance(obj, list):
    return [convert_to_native(x) for x in obj]
  elif isinstance(obj, (np.integer, np.int64, np.int32)):
    return int(obj)
  elif isinstance(obj, (np.floating, np.float64, np.float32)):
    return float(obj)
  elif isinstance(obj, np.ndarray):
    return obj.tolist()
  else:
    return obj
