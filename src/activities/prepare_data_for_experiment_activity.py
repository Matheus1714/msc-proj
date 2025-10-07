from temporalio import activity

from src.default_types import PrepareDataForExperimentIn, PrepareDataForExperimentOut

@activity.defn
def prepare_data_for_experiment_activity(data: PrepareDataForExperimentIn) -> PrepareDataForExperimentOut:
  return PrepareDataForExperimentOut(
    input_data_path="",
    ground_truth_path="",
  )
