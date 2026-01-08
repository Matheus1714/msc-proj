import os
import pandas as pd
from typing import List, Tuple

def create_subsets_from_csv(
  input_file_path: str,
  output_dir: str,
  percentages: List[int] = [],
  max_rows: int = 2000
) -> List[Tuple[int, str, int]]:
  os.makedirs(output_dir, exist_ok=True)
  
  df = pd.read_csv(input_file_path)
  limit = len(df)

  shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True).head(max_rows)
  total_rows = min(limit, max_rows)
  
  data_subsets = []
  
  for pct in percentages:
    target_rows = int((pct / 100) * total_rows)

    subset_df = shuffled_df.head(target_rows)

    filename = f"academic_works_{pct}pct_{len(subset_df)}rows.csv"
    filepath = os.path.join(output_dir, filename)
    
    subset_df.to_csv(filepath, index=False)
    
    data_subsets.append((pct, filepath, len(subset_df)))
  
  return data_subsets
