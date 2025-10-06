from typing import (
  TypedDict,
  List,
  Dict,
  NewType,
  Tuple,
)
from datetime import datetime

Id = NewType("Id", str)

class AcademicWork(TypedDict):
  id: str
  title: str
  abstract: str
  keywords: List[str]
  included: bool
  db_source: str
  created_at: datetime
  updated_at: datetime

class InputData(TypedDict):
  title: str
  keywords: List[str]
  keywords: List[str]

class GroundTruth(TypedDict):
  included: bool

class DataItem(TypedDict):
  id: str
  input: InputData
  ground_truth: GroundTruth

class PredictionIn(TypedDict):
  data: List[DataItem]

class PredictionOut(TypedDict):
  predicted_labels: Dict[Id, bool]

class EvaluationData(TypedDict):
  true_positives: int
  false_positives: int
  true_negatives: int
  false_negatives: int
  total_positives: int
  total_negatives: int
  precision: float
  recall: float
  f2_score: float
  wss95: float

class SimulationConfig(TypedDict):
  classification_strategy: str
  tokenizer_strategy: str
  version: str
  dataset_version: str

class SimulationRun(TypedDict):
  id: str
  input: PredictionIn
  output: PredictionOut
  evaluation: EvaluationData
  config: SimulationConfig
  created_at: datetime
  updated_at: datetime

class MLSimulationWorkflowIn(TypedDict):
  ...

class MLSimulationWorkflowOut(TypedDict):
  ...

class DataPreprocessingWorkflowIn(TypedDict):
  source_files: List[Tuple[str, str]]
  output_path: str

class DataPreprocessingWorkflowOut(TypedDict):
  total_processed_works: int

class ProcessGoogleDriveFileIn(TypedDict):
  file_name: str
  file_id: str

class ProcessGoogleDriveFileOut(TypedDict):
  file_name: str

class MergeProcessedDataIn(TypedDict):
  all_processed_files: List[str]
  output_path: str

class MergeProcessedDataOut(TypedDict):
  total_processed_works: int
