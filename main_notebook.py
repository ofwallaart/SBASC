# Databricks notebook source
from hydra import compose, initialize

from models.BERT_baseline.main import BertBaseline
from models.SBASC.main import SBASC
from models.WBASC.main import WBASC

from run_hyperparameter import run_trials
from run import store_run_results

# COMMAND ----------

# DBTITLE 1,Do a single run for creating labels
with initialize(config_path="conf"):
  cfg = compose("config.yaml", overrides=[f'domain=restaurant3', 'model=WBASC', 'environment=databricks', 'ablation=none'])
  results = WBASC(cfg)(load=True, evaluate=False)

# COMMAND ----------

# DBTITLE 1,Hyperparameter tuning
domains = ['restaurant3']
for domain in domains:
  with initialize(config_path="conf"):
    cfg = compose("config.yaml", overrides=[f'domain={domain}', 'model=WBASC', 'environment=databricks'])
    models = [WBASC(cfg)]
    run_trials(models, cfg, 60)

# COMMAND ----------

import pickle

trials = pickle.load(open("/dbfs/FileStore/kto/results/restaurant-3/WBASC/results.pkl", "rb"))
best = trials.best_trial['result']
print(best)
for trial in trials.trials:
  print(trial['result']['loss'])
  print(trial['result']['accuracy'])
  print(trial['result']['space'])
  print()

# COMMAND ----------

# DBTITLE 1,Run 5 times for creating results
from tqdm import trange

domains = ['laptop']
ablations = ['woDL', 'woSBERT']
for domain in domains:
  for ablation in ablations:
    for i in trange(5):
      with initialize(config_path="conf"):
        cfg = compose("config.yaml", overrides=[f'domain={domain}', 'model=WBASC', 'environment=databricks', f'ablation={ablation}'])
        results = WBASC(cfg)(load=True, evaluate=False)
        store_run_results(results, cfg.result_path_mapper, cfg.model.name, cfg.ablation.name)
