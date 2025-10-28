from enum import Enum

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
GLOVE_EMBEDDINGS_PATH = "data/glove_embeddings.npy"
TOKENIZED_DATA_PATH = "data/tokenized_data.csv"
WORD_INDEX_PATH = "data/word_index.json"

X_SEQ_PATH = "data/x_seq.npy"
Y_PATH = "data/y.npy"

X_TRAIN_PATH = "data/x_train.npy"
X_VAL_PATH = "data/x_val.npy"
X_TEST_PATH = "data/x_test.npy"
Y_TRAIN_PATH = "data/y_train.npy"
Y_VAL_PATH = "data/y_val.npy"
Y_TEST_PATH = "data/y_test.npy"

PREPARED_DATA_PATH = "data/prepared_data.csv"


# ExperimentBiLSTMWithGlove

# ExperimentBiLSTMWithGloveAndAttention

# ExperimentLSTMWithGlove

# ExperimentLSTMWithGloveAndAttention

