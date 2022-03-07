# Databricks notebook source
from models.BERT_baseline.main import BertBaseline
from models.SBASC.main import SBASC
from models.WBASC.main import WBASC

from run_hyperparameter import run_trials

# COMMAND ----------

# DBTITLE 1,Do a single run for creating labels
SBASC()(load=True)

# COMMAND ----------

models = [SBASC()]
run_trials(models, 60)

# COMMAND ----------

import pickle

trials = pickle.load(open("/dbfs/FileStore/kto/results/restaurant-3/SBASC/results.pkl", "rb"))
best = trials.best_trial['result']
print(best['space'])
