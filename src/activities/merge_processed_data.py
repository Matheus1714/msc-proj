import pandas as pd
from temporalio import activity

from src.default_types import MergeProcessedDataIn, MergeProcessedDataOut


@activity.defn
async def merge_processed_data(data: MergeProcessedDataIn) -> MergeProcessedDataOut:
  all_data = []
  for file_name in data.all_processed_files:
    df = pd.read_csv(file_name)
    all_data.extend(df.to_dict(orient='records'))
  
  df_final = pd.DataFrame(all_data)
  
  df_final.to_csv(data.output_path, index=False)
  
  activity.logger.info(f"✅ {len(df_final)} trabalhos salvos em {data.output_path}")
  return len(df_final)
