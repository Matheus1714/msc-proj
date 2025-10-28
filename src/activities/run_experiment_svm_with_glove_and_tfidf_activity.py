import pandas as pd
import numpy as np
from typing import Tuple

from temporalio import activity
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

@dataclass
class RunExperimentSVMWithGloveAndTFIDFIn:
  input_data_path: str
  y_path: str
  random_state: int
  max_iter: int
  ngram_range: Tuple[int, int]

@dataclass
class RunExperimentSVMWithGloveAndTFIDFOut:
  ...

@activity.defn
async def run_experiment_svm_with_glove_and_tfidf_activity(data: RunExperimentSVMWithGloveAndTFIDFIn) -> RunExperimentSVMWithGloveAndTFIDFOut:
  df = pd.read_csv(data.input_data_path)
  y = np.load(data.y_path)

  vect = TfidfVectorizer(ngram_range=data.ngram_range)
  X_tfidf = vect.fit_transform(df["text"].to_list())
  cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=data.random_state)
  clf = SGDClassifier(loss='hinge', penalty='l2', max_iter=data.max_iter, random_state=data.random_state)
  y_scores = cross_val_predict(clf, X_tfidf, y, cv=cv, method='decision_function')

  y_true_scores = y

  prec, rec, thresh = precision_recall_curve(y_true_scores, y_scores)

  best_thresh = next((t for p, r, t in zip(prec, rec, thresh) if r >= 0.95), 0.5)

  y_pred = (y_scores >= best_thresh).astype(int)
  tn, fp, fn, tp = confusion_matrix(y_true_scores, y_pred).ravel()

  N = tn + fp
  P = tp / (tp + fp) if (tp + fp) else 0
  R = tp / (tp + fn) if (tp + fn) else 0
  F2 = (5 * P * R) / (4 * P + R) if (P + R) else 0
  WSS95 = (N - fp)/N - (1-0.95) if N else 0

  return RunExperimentSVMWithGloveAndTFIDFOut()
