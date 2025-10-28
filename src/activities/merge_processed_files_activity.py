import os
import pandas as pd
from temporalio import activity
from dataclasses import dataclass
from typing import List

@dataclass
class MergeProcessedDataIn:
  all_processed_files: List[str]
  output_path: str

@activity.defn
async def merge_processed_files_activity(data: MergeProcessedDataIn) -> int:
  all_data = []
  for file_name in data.all_processed_files:
    df = pd.read_csv(file_name)
    all_data.extend(df.to_dict(orient='records'))
    os.remove(file_name)

  df_final = pd.DataFrame(all_data)
  
  df_final.to_csv(data.output_path, index=False)
  
  activity.logger.info(f"✅ {len(df_final)} trabalhos salvos em {data.output_path}")
  return len(df_final)
