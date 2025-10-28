import pandas as pd
import numpy as np
from typing import Tuple

from temporalio import activity
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.utils.calculate_metrics import calculate_metrics, EvaluationData

@dataclass
class RunExperimentSVMWithGloveAndTFIDFIn:
  input_data_path: str
  y_path: str
  random_state: int
  max_iter: int
  ngram_range: Tuple[int, int]

@dataclass
class RunExperimentSVMWithGloveAndTFIDFOut:
  metrics: EvaluationData

@activity.defn
async def run_experiment_svm_with_glove_and_tfidf_activity(data: RunExperimentSVMWithGloveAndTFIDFIn) -> RunExperimentSVMWithGloveAndTFIDFOut:
  df = pd.read_csv(data.input_data_path)
  y = np.load(data.y_path)

  vect = TfidfVectorizer(ngram_range=data.ngram_range)
  X_tfidf = vect.fit_transform(df["text"].to_list())
  cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=data.random_state)
  clf = SGDClassifier(loss='hinge', penalty='l2', max_iter=data.max_iter, random_state=data.random_state)
  y_scores = cross_val_predict(clf, X_tfidf, y, cv=cv, method='decision_function')

  metrics = calculate_metrics(y, y_scores)

  return RunExperimentSVMWithGloveAndTFIDFOut(metrics=metrics)
