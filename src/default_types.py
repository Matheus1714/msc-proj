from typing import (
  TypedDict,
  List,
  Dict,
  NewType,
  Tuple,
  Any,
  Optional,
)
from datetime import datetime
from dataclasses import dataclass

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

@dataclass
class DataPreprocessingWorkflowIn:
  output_path: str

@dataclass
class DataPreprocessingWorkflowOut:
  total_processed_works: int

@dataclass
class ProcessGoogleDriveFileIn:
  file_name: str
  file_path: str

@dataclass
class ProcessGoogleDriveFileOut:
  file_path: str

@dataclass
class MergeProcessedDataIn:
  all_processed_files: List[str]
  output_path: str

@dataclass
class PrepareDataForExperimentIn:
  id: str
  file_path: str

@dataclass
class PrepareDataForExperimentOut:
  input_data_path: str
  ground_truth_path: str

@dataclass
class TokenizeSharedWorkflowIn:
  file_path: str
  strategy: str

@dataclass
class TokenizeSharedWorkflowOut:
  tokenized_data_path: str

@dataclass
class SimulateModelWorkflowIn:
  file_path: str
  strategy: str

@dataclass
class SimulateModelWorkflowOut:
  result: str

@dataclass
class ModelConfig:
  name: str
  type: str  # "svm", "random_forest", etc.
  hyperparameters: Dict[str, Any]

@dataclass
class ExperimentWorkflowIn:
  dataset_id: str
  model_config: ModelConfig
  tokenizer_strategy: str
  model_path: Optional[str] = None  # Se fornecido, não treina novamente

@dataclass
class ExperimentWorkflowOut:
  model_path: str
  validation_metrics_path: str
  production_metrics_path: str
  final_report_path: str

@dataclass
class TrainModelIn:
  model_config: ModelConfig
  training_data_path: str
  model_output_path: str

@dataclass
class TrainModelOut:
  model_path: str
  training_metrics: Dict[str, float]

@dataclass
class ValidateModelIn:
  model_path: str
  validation_data_path: str
  metrics_output_path: str

@dataclass
class ValidateModelOut:
  validation_metrics_path: str
  metrics: Dict[str, float]

@dataclass
class RunProductionInferenceIn:
  model_path: str
  production_data_path: str
  predictions_output_path: str

@dataclass
class RunProductionInferenceOut:
  predictions_path: str
  production_metrics: Dict[str, float]

@dataclass
class AggregateResultsIn:
  validation_metrics_path: str
  production_metrics_path: str
  report_output_path: str

@dataclass
class AggregateResultsOut:
  final_report_path: str
