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
  cfg = compose("config.yaml", overrides=[f'domain=supermarket', 'model=WBASC', 'environment=databricks', 'ablation=none'])
  results = WBASC(cfg)(load=False, evaluate=True)

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = spark.read.options(header='True').option("quote", "\"").option("escape", "\"").csv(r'dbfs:/FileStore/kto/supermarket/predictions.csv')

classes = df.select('actual category').distinct().rdd.flatMap(lambda x: x).collect()

cm = confusion_matrix(df.select('actual category').rdd.flatMap(lambda x: x).collect(), df.select('predicted category').rdd.flatMap(lambda x: x).collect(), labels=classes)
cmp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

fig, ax = plt.subplots(figsize=(15,15))
cmp.plot(ax=ax, xticks_rotation='vertical')

# COMMAND ----------

display(df)

# COMMAND ----------

# DBTITLE 1,Hyperparameter tuning
domains = ['supermarket']
for domain in domains:
  with initialize(config_path="conf"):
    cfg = compose("config.yaml", overrides=[f'domain={domain}', 'model=WBASC', 'environment=databricks'])
    models = [WBASC(cfg)]
    run_trials(models, cfg, 40)

# COMMAND ----------

import pickle

trials = pickle.load(open("/dbfs/FileStore/kto/results/supermarket/WBASC/results.pkl", "rb"))
best = trials.best_trial['result']
print(best)
for i, trial in enumerate(trials.trials):
  print(i)
  print(trial['result']['loss'])
  print(trial['result']['accuracy'])
  print(trial['result']['space'])
  print()

# COMMAND ----------

# DBTITLE 1,Run ablations for creating results
from tqdm import trange

domains = ['supermarket']
ablations = ['none','woDL', 'woDK', 'woSBERT']
for domain in domains:
  for ablation in ablations:
      with initialize(config_path="conf"):
        cfg = compose("config.yaml", overrides=[f'domain={domain}', 'model=WBASC', 'environment=databricks', f'ablation={ablation}'])
        results = WBASC(cfg)(load=True, evaluate=False)
        store_run_results(results, cfg.result_path_mapper, cfg.model.name, cfg.ablation.name)

# COMMAND ----------

from pyspark.sql.functions import avg, col, lit, when

df = spark.read.format('csv').options(header='true', inferSchema='true').load('dbfs:/FileStore/kto/results/restaurant-nl/SBASC/results.csv')
display(df.sort("ablation","type")) 
# display(df[df['timestamp'] > '2022-03-16'].sort("ablation","type")) 
