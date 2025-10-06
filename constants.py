from enum import Enum

GOOGLE_DRIVE_FILES_ID = [
  ("UrinaryIncontinence.csv", "1CQXJvSpA7KVeGImypLSBPerPlzd-7Osp"),
  ("Hall_2012.csv", "1HUc4UxTQ3_4SaPz2FvXUkuB49Y5YNB1p"),
  ("ProtonPumpInhibitors.csv", "1t8MsIwLdjF3LsrFRjy00tvU52gVsG4ld"),
  ("van_de_Schoot_2017.csv", "1BFQTrOg-FjTK5gtw-1pUBpqFvrLRJoij"),
  ("SkeletalMuscleRelaxants.csv", "1LXOdKDGWG7ZE_nsGJFn0VfKOkp-3yuBu"),
  ("Kwok_2020.csv", "1cnZoP28syi7N9bLeerk8f234C4E53KqE"),
  ("Nagtegaal_2019.csv", "1P5hMvAiYomEwp2TY9TPOb2B8Ih9dX6Qc"),
  ("Wolters_2018.csv", "1f75DjbnWYiJJAumzOWNCFTd9oLZjb68S"),
  ("Kitchenham_2010.csv", "1A5Y1Nu3udcsqHUN6jOeVBKXjtSeyaQVb"),
  ("Radjenovic_2013.csv", "14J4yjyjy0u4rvNyRZ7nUxOPWU0dtUm86"),
  ("Triptans.csv", "1wDQzkzP4z6p8IT5egxszJZQmtuHJ-MJD"),
  ("Wahono_2015.csv", "1t72QNc3_miWsS_ht2Kxc40PnE9Lp9ZqC"),
  ("Statins.csv", "1mLg5o9M1HDcjyOpZUXxs2EF0OWvDuov-"),
  ("van_Dis_2020.csv", "1JTCX0gQ2KIwnKfud_7LOV11NkaqzmJHo"),
  ("ADHD.csv", "1FMGWV80MI1Pq8_a_GuIxO_5xZrw71ajm"),
  ("ACEInhibitors.csv", "1X7S4JT7wx-D7vxwmgBRWowyZ-PJoDmkv"),
  ("Opiods.csv", "1xf-HecsXREPHwFr-dClt3nk2G7yqyxms"),
  ("AtypicalAntipsychotics.csv", "1qgguRWxDqqEk3Go3wahopei0RSvMkt6q"),
  ("CalciumChannelBlockers.csv", "1EmT3H3emE9KlCAaYkMEd-F71vgQbBy0q"),
  ("BetaBlockers.csv", "1h7gRtBf6YqIlwu4dYVCNbZgQWl3s7Zvm"),
  ("NSAIDS.csv", "137mglPeSXUr5wVkzO3QmzKWVsgz0B9Iq"),
  ("Bannach-Brown_2019.csv", "1Xid0OTXSycHRTR7CzpZ2BYjScBrttPQS"),
  ("Appenzeller-Herzog_2020.csv", "1sPxDAQtmfB3nOjeJWwsyLxcFQrbYs3R3"),
  ("OralHypoglycemics.csv", "188Yb5svRlphVuVwUudKWq3TsryWdJ2Cp"),
  ("Antihistamines.csv", "1FSa6URUSiqbzuDq6cwVOK5IwdCauQkXF"),
  ("Bos_2018.csv", "1H5yBNB_BtuHtxgcDLI7q1YfMiGwAxu-I"),
  ("Estrogens.csv", "1ASCDXZ1yP0GTaQYdbBvWRxmFdXNw-O1f"),
]

class WorflowTaskQueue(Enum):
  ML_TASK_QUEUE = "ml-task-queue"
