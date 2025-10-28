from temporalio import activity
import pandas as pd

from dataclasses import dataclass

from constants import PREPARED_DATA_PATH

@dataclass
class PrepareDataForExperimentIn:
  input_data_path: str
  random_state: int

@dataclass
class PrepareDataForExperimentOut:
  output_data_path: str

@activity.defn
async def prepare_data_for_experiment_activity(data: PrepareDataForExperimentIn) -> PrepareDataForExperimentOut:
  df = pd.read_csv(data.input_data_path)
  df = df.head(50).copy() ## TODO: Remover daqui quando for para produção
  df = df.sample(frac=1, random_state=data.random_state).reset_index(drop=True)

  df["included"] = df["included"].astype(bool)
  df["text"] = (df["title"].fillna("") + " " +
                df["keywords"].fillna("") + " " +
                df["abstract"].fillna(""))
  
  df.to_csv(PREPARED_DATA_PATH, index=False)

  return PrepareDataForExperimentOut(
    output_data_path=PREPARED_DATA_PATH,
  )
