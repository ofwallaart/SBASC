# Databricks notebook source
from hydra import compose, initialize

from models.BERT_baseline.main import BertBaseline
from models.SBASC.main import SBASC
from models.WBASC.main import WBASC

from run_hyperparameter import run_trials
from run import store_run_results

# COMMAND ----------

# DBTITLE 1,Do a single run for creating labels
from tqdm import trange

domains = ['restaurant3', 'restaurant5', 'laptop']
for domain in domains:
  for i in trange(5):
    with initialize(config_path="conf"):
      cfg = compose("config.yaml", overrides=[f'domain={domain}', 'model=SBASC', 'environment=databricks', 'ablation=woSBERT'])
      results = SBASC(cfg)(load=True)
      store_run_results(results, cfg.result_path_mapper, cfg.model.name, cfg.ablation.name)

# COMMAND ----------

with initialize(config_path="conf"):
      cfg = compose("config.yaml", overrides=[f'domain=restaurant3', 'model=SBASC', 'environment=databricks', 'ablation=woSBERT'])
      results = SBASC(cfg)(load=True)

# COMMAND ----------

from pyspark.sql.functions import avg, col, lit, when

df = spark.read.format('csv').options(header='true', inferSchema='true').load('dbfs:/FileStore/kto/results/restaurant-3/SBASC/results.csv')
# df = df.withColumn('ablation', when(col('ablation') == 'WithoutSBERT', lit('none')).otherwise(col('ablation')))

display(df.groupBy('type', 'ablation').agg(avg('accuracy'), avg('precision'), avg('recall'), avg('f1-score'))) 

# df.toPandas().to_csv('/dbfs/FileStore/kto/results/laptop/SBASC/results.csv', mode='w')

# COMMAND ----------

# DBTITLE 1,Hyperparameter tuning
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
