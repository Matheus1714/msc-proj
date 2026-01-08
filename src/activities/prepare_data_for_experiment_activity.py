from temporalio import activity
import pandas as pd
import os

from dataclasses import dataclass

@dataclass
class PrepareDataForExperimentIn:
  input_data_path: str
  output_data_path: str
  random_state: int

@dataclass
class PrepareDataForExperimentOut:
  output_data_path: str

@activity.defn
async def prepare_data_for_experiment_activity(data: PrepareDataForExperimentIn) -> PrepareDataForExperimentOut:
  os.makedirs(os.path.dirname(data.output_data_path), exist_ok=True)
  
  df = pd.read_csv(data.input_data_path)
  df = df.sample(frac=1, random_state=data.random_state).reset_index(drop=True)

  df["included"] = df["included"].astype(bool)
  df["text"] = (df["title"].fillna("") + " " +
                df["keywords"].fillna("") + " " +
                df["abstract"].fillna(""))
  # df["text"] = df["abstract"].fillna("")
  
  df.to_csv(data.output_data_path, index=False)

  return PrepareDataForExperimentOut(
    output_data_path=data.output_data_path,
  )
