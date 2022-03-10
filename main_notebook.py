# Databricks notebook source
from hydra import compose, initialize

from models.BERT_baseline.main import BertBaseline
from models.SBASC.main import SBASC
from models.WBASC.main import WBASC

from run_hyperparameter import run_trials

# COMMAND ----------

# DBTITLE 1,Do a single run for creating labels
with initialize(config_path="conf"):
  cfg = compose("config.yaml", overrides=['domain=laptop', 'model=SBASC', 'environment=databricks'])
  SBASC(cfg)(load=True)

# COMMAND ----------

with initialize(config_path="conf"):
    cfg = compose("config.yaml", overrides=['domain=laptop', 'model=SBASC', 'environment=databricks'])
    models = [SBASC(cfg)]
    run_trials(models, cfg, 60)

# COMMAND ----------

import pickle

trials = pickle.load(open("/dbfs/FileStore/kto/results/laptop/SBASC/results.pkl", "rb"))
best = trials.best_trial['result']
print(best)
for trial in trials.trials:
  print(trial['result']['loss'])
  print(trial['result']['accuracy'])
  print(trial['result']['space'])
  print()
