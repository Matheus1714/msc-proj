import numpy as np
from typing import TypedDict

from sklearn.metrics import precision_recall_curve, confusion_matrix

class EvaluationData(TypedDict):
  true_positives: int
  false_positives: int
  true_negatives: int
  false_negatives: int
  total_positives: int
  total_negatives: int
  precision: float
  recall: float
  f1_score: float
  f2_score: float
  wss95: float
  accuracy: float

def calculate_metrics(
  y_true: np.ndarray, y: np.ndarray
) -> EvaluationData:
  prec, rec, thresh = precision_recall_curve(y_true, y)

  # Encontrar o MAIOR threshold que ainda atinge recall >= 0.95
  # Os thresholds estão em ordem decrescente, então pegamos o último que satisfaz
  valid_thresholds = [t for p, r, t in zip(prec, rec, thresh) if r >= 0.95]
  best_thresh = valid_thresholds[-1] if valid_thresholds else 0.5

  y_pred = (y >= best_thresh).astype(int)
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

  N = tn + fp + fn + tp
  P = tp / (tp + fp) if (tp + fp) else 0
  R = tp / (tp + fn) if (tp + fn) else 0
  F1 = 2 * P * R / (P + R) if (P + R) else 0
  F2 = (5 * P * R) / (4 * P + R) if (P + R) else 0
  WSS95 = (tn + fn)/N - (1-0.95) if N else 0
  accuracy = (tp + tn) / N if N else 0

  return EvaluationData(
    true_positives=tp,
    false_positives=fp,
    true_negatives=tn,
    false_negatives=fn,
    total_positives=tp + fn,
    total_negatives=tn + fp,
    precision=P,
    recall=R,
    f1_score=F1,
    f2_score=F2,
    wss95=WSS95,
    accuracy=accuracy
  )
