import pandas as pd
import numpy as np
from typing import Tuple

from temporalio import activity
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.utils.calculate_metrics import calculate_metrics, EvaluationData
from src.utils.convert_to_native import convert_to_native

@dataclass
class RunExperimentSVMWithGloveAndTFIDFIn:
  input_data_path: str
  y_path: str
  random_state: int
  max_iter: int
  ngram_range: Tuple[int, int]
  class_weight_0: float = 1.0
  class_weight_1: float = 1.0

@dataclass
class RunExperimentSVMWithGloveAndTFIDFOut:
  metrics: EvaluationData

@activity.defn
async def run_experiment_svm_with_glove_and_tfidf_activity(data: RunExperimentSVMWithGloveAndTFIDFIn) -> RunExperimentSVMWithGloveAndTFIDFOut:
  df = pd.read_csv(data.input_data_path)
  y = np.load(data.y_path)

  if len(y) != len(df):
    raise ValueError(f"Mismatch between y length ({len(y)}) and dataframe length ({len(df)})")

  y = y.astype(int)

  vect = TfidfVectorizer(ngram_range=data.ngram_range)
  X_tfidf = vect.fit_transform(df["text"].to_list())
  cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=data.random_state)
  clf = SGDClassifier(
    loss='hinge', 
    penalty='l2', 
    max_iter=data.max_iter, 
    random_state=data.random_state,
    class_weight={0: data.class_weight_0, 1: data.class_weight_1}
  )
  y_scores = cross_val_predict(clf, X_tfidf, y, cv=cv, method='decision_function')

  metrics = calculate_metrics(y, y_scores)

  return RunExperimentSVMWithGloveAndTFIDFOut(metrics=convert_to_native(metrics))

