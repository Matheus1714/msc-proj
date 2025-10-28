from enum import Enum
from dataclasses import dataclass

SOURCE_INPUT_FILES = [
  ("UrinaryIncontinence.csv", "/Users/matheusmota/src/github/msc/systematic-review-datasets/datasets/Cohen_2006/output/local/UrinaryIncontinence.csv"),
  ("Hall_2012.csv", "/Users/matheusmota/src/github/msc/systematic-review-datasets/datasets/Hall_Wahono_Radjenovic_Kitchenham/output/Hall_2012.csv"),
  ("ProtonPumpInhibitors.csv", "/Users/matheusmota/src/github/msc/systematic-review-datasets/datasets/Cohen_2006/output/local/ProtonPumpInhibitors.csv"),
  ("van_de_Schoot_2017.csv", "/Users/matheusmota/src/github/msc/systematic-review-datasets/datasets/van_de_Schoot_2017/output/van_de_Schoot_2017.csv"),
  ("SkeletalMuscleRelaxants.csv", "/Users/matheusmota/src/github/msc/systematic-review-datasets/datasets/Cohen_2006/output/local/SkeletalMuscleRelaxants.csv"),
  ("Kwok_2020.csv", "/Users/matheusmota/src/github/msc/systematic-review-datasets/datasets/Kwok_2020/output/Kwok_2020.csv"),
  ("Nagtegaal_2019.csv", "/Users/matheusmota/src/github/msc/systematic-review-datasets/datasets/Nagtegaal_2019/output/Nagtegaal_2019.csv"),
  ("Wolters_2018.csv", "/Users/matheusmota/src/github/msc/systematic-review-datasets/datasets/Wolters_2018/output/Wolters_2018.csv"),
  ("Kitchenham_2010.csv", "/Users/matheusmota/src/github/msc/systematic-review-datasets/datasets/Hall_Wahono_Radjenovic_Kitchenham/output/Kitchenham_2010.csv"),
  ("Radjenovic_2013.csv", "/Users/matheusmota/src/github/msc/systematic-review-datasets/datasets/Hall_Wahono_Radjenovic_Kitchenham/output/Radjenovic_2013.csv"),
  ("Triptans.csv", "/Users/matheusmota/src/github/msc/systematic-review-datasets/datasets/Cohen_2006/output/local/Triptans.csv"),
  ("Wahono_2015.csv", "/Users/matheusmota/src/github/msc/systematic-review-datasets/datasets/Hall_Wahono_Radjenovic_Kitchenham/output/Wahono_2015.csv"),
  ("Statins.csv", "/Users/matheusmota/src/github/msc/systematic-review-datasets/datasets/Cohen_2006/output/local/Statins.csv"),
  ("van_Dis_2020.csv", "/Users/matheusmota/src/github/msc/systematic-review-datasets/datasets/van_Dis_2020/output/van_Dis_2020.csv"),
  ("ADHD.csv", "/Users/matheusmota/src/github/msc/systematic-review-datasets/datasets/Cohen_2006/output/local/ADHD.csv"),
  ("ACEInhibitors.csv", "/Users/matheusmota/src/github/msc/systematic-review-datasets/datasets/Cohen_2006/output/local/ACEInhibitors.csv"),
  ("Opiods.csv", "/Users/matheusmota/src/github/msc/systematic-review-datasets/datasets/Cohen_2006/output/local/Opiods.csv"),
  ("AtypicalAntipsychotics.csv", "/Users/matheusmota/src/github/msc/systematic-review-datasets/datasets/Cohen_2006/output/local/AtypicalAntipsychotics.csv"),
  ("CalciumChannelBlockers.csv", "/Users/matheusmota/src/github/msc/systematic-review-datasets/datasets/Cohen_2006/output/local/CalciumChannelBlockers.csv"),
  ("BetaBlockers.csv", "/Users/matheusmota/src/github/msc/systematic-review-datasets/datasets/Cohen_2006/output/local/BetaBlockers.csv"),
  ("NSAIDS.csv", "/Users/matheusmota/src/github/msc/systematic-review-datasets/datasets/Cohen_2006/output/local/NSAIDS.csv"),
  ("Bannach-Brown_2019.csv", "/Users/matheusmota/src/github/msc/systematic-review-datasets/datasets/Bannach-Brown_2019/output/Bannach-Brown_2019.csv"),
  ("Appenzeller-Herzog_2020.csv", "/Users/matheusmota/src/github/msc/systematic-review-datasets/datasets/Appenzeller-Herzog_2020/output/Appenzeller-Herzog_2020.csv"),
  ("OralHypoglycemics.csv", "/Users/matheusmota/src/github/msc/systematic-review-datasets/datasets/Cohen_2006/output/local/OralHypoglycemics.csv"),
  ("Antihistamines.csv", "/Users/matheusmota/src/github/msc/systematic-review-datasets/datasets/Cohen_2006/output/local/Antihistamines.csv"),
  ("Bos_2018.csv", "/Users/matheusmota/src/github/msc/systematic-review-datasets/datasets/Bos_2018/output/Bos_2018.csv"),
  ("Estrogens.csv", "/Users/matheusmota/src/github/msc/systematic-review-datasets/datasets/Cohen_2006/output/local/Estrogens.csv"),
]
class WorflowTaskQueue(Enum):
  ML_TASK_QUEUE = "ml-task-queue"

class ExperimentType(Enum):
  SVM_WITH_SGD_AND_TF_IDF = "svm_with_sgd_and_tf_idf"
  BI_LSTM_WITH_GLOVE = "bi_lstm_with_glove"
  BI_LSTM_WITH_GLOVE_AND_ATTENTION = "bi_lstm_with_glove_and_attention"
  LSTM_WITH_GLOVE = "lstm_with_glove"
  LSTM_WITH_GLOVE_AND_ATTENTION = "lstm_with_glove_and_attention"

GLOVE_6B_300D_FILE_PATH = "data/word_vectors/glove/glove.6B.300d.txt"

@dataclass
class ExperimentConfig:
  experiment_id: str
  base_dir: str
  prepared_data_path: str
  tokenized_data_path: str
  word_index_path: str
  glove_embeddings_path: str
  x_seq_path: str
  y_path: str
  x_train_path: str
  x_val_path: str
  x_test_path: str
  y_train_path: str
  y_val_path: str
  y_test_path: str
  results_file_path: str
  machine_specs_file_path: str
  
  @classmethod
  def create(cls, experiment_id: str = None):
    if experiment_id is None:
      from datetime import datetime
      experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    base_dir = f"data/experiments/{experiment_id}"
    
    return cls(
      experiment_id=experiment_id,
      base_dir=base_dir,
      prepared_data_path=f"{base_dir}/prepared_data.csv",
      tokenized_data_path=f"{base_dir}/tokenized_data.csv",
      word_index_path=f"{base_dir}/word_index.json",
      glove_embeddings_path=f"{base_dir}/glove_embeddings.npy",
      x_seq_path=f"{base_dir}/x_seq.npy",
      y_path=f"{base_dir}/y.npy",
      x_train_path=f"{base_dir}/x_train.npy",
      x_val_path=f"{base_dir}/x_val.npy",
      x_test_path=f"{base_dir}/x_test.npy",
      y_train_path=f"{base_dir}/y_train.npy",
      y_val_path=f"{base_dir}/y_val.npy",
      y_test_path=f"{base_dir}/y_test.npy",
      results_file_path=f"{base_dir}/experiment_results.csv",
      machine_specs_file_path=f"{base_dir}/machine_specs.txt"
    )
  
  def create_directories(self):
    import os
    os.makedirs(self.base_dir, exist_ok=True)
    return self.base_dir
