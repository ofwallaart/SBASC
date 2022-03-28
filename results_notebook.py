# Databricks notebook source
from hydra import compose, initialize

from models.BERT_baseline.main import BertBaseline
from models.SBASC.main import SBASC
from models.WBASC.main import WBASC

from run_hyperparameter import run_trials
from run import store_run_results

ablations = ['woSBERT']
for ablation in ablations:
  with initialize(config_path="conf"):
    cfg = compose("config.yaml", overrides=[f'domain=restaurantnl', 'model=WBASC', 'environment=databricks', f'ablation={ablation}'])
    print(cfg)
    results = WBASC(cfg)(load=True, evaluate=True)
    store_run_results(results, cfg.result_path_mapper, cfg.model.name, cfg.ablation.name)

# COMMAND ----------

from pyspark.sql.functions import avg, col, lit, when

df = spark.read.format('csv').options(header='true', inferSchema='true').load('dbfs:/FileStore/kto/results/restaurant-3/WBASC/results.csv')

display(df.sort("ablation","type")) 
# display(df.groupBy('type', 'ablation').agg(avg('accuracy'), avg('precision'), avg('recall'), avg('f1-score')).sort("ablation","type"))

# COMMAND ----------


df = spark.read.format('csv').options(header='true', inferSchema='true').load('dbfs:/FileStore/kto/restaurant-nl/train.txt')
print(df.count())

# COMMAND ----------

df = spark.read.format('csv').options(header='true', inferSchema='true').load('dbfs:/FileStore/kto/results/restaurant-nl/SBASC/results.csv')

display(df.sort("ablation","type"))

# COMMAND ----------

import pickle

trials = pickle.load(open("/dbfs/FileStore/kto/results/laptop/WBASC/results.pkl", "rb"))
best = trials.best_trial['result']
print(best)
for trial in trials.trials:
  print(trial['result']['loss'])
  print(trial['result']['accuracy'])
  print(trial['result']['space'])
  print()

# COMMAND ----------

from glob import glob
import pandas as pd
import json
from hydra import compose, initialize

f_names = [
  {'model':'SBASC', 'data':'/dbfs/FileStore/kto/results/restaurant-3/SBASC/run_20220316_184245_none.json'},
  {'model':'SBASCwoSBERT', 'data':'/dbfs/FileStore/kto/results/restaurant-3/SBASC/run_20220317_083011_WithoutSBERT.json'},
  {'model':'SBASCwoDK', 'data':'/dbfs/FileStore/kto/results/restaurant-3/SBASC/run_20220316_184902_WithoutDomainKnowledge.json'},
  {'model':'SBASCwoDL', 'data':'/dbfs/FileStore/kto/results/restaurant-3/SBASC/run_20220318_150055_WithoutDeepLearning.json'},
  {'model':'WBASC', 'data':'/dbfs/FileStore/kto/results/restaurant-3/WBASC/run_20220315_121528_none.json'},
          ]

with initialize(config_path="conf"):
  cfg = compose("config.yaml", overrides=[f'domain=restaurant3', 'model=SBASC', 'environment=databricks', f'ablation=none'])
  
  aspects = cfg.domain.aspect_category_mapper
  polarities = cfg.domain.sentiment_category_mapper
  
  df = pd.read_csv(f'/dbfs/FileStore/kto/{cfg.domain.name}/test.txt', sep='\t', header=None, index_col=0).rename(columns={1: "aspect", 2: "sentiment", 3:"sentence"})
  df['aspect'] = df['aspect'].apply(lambda x: aspects[int(x)])
  df['sentiment'] = df['sentiment'].apply(lambda x: polarities[int(x)])
  
  for f_name in f_names:
    with open(f_name['data'], 'r') as f:
      data = json.load(f)
      model = f_name['model']
      if model == 'SBASCwoDL':
        df[f"aspect_{model}"] = data["predicted_asp"]
        df[f"sentiment_{model}"] = data["predicted_pol"]
        df[f"aspect_{model}"] = df[f"aspect_{model}"].apply(lambda x: aspects[int(x)])
        df[f"sentiment_{model}"] = df[f"sentiment_{model}"].apply(lambda x: polarities[int(x)])
      else:
        df[f"aspect_{model}"] = data["predicted_asp"]
        df[f"sentiment_{model}"] = data["predicted_pol"]
      
      df[f"{model}"] = df[f"aspect_{model}"].astype(str)+ " " + df[f"sentiment_{model}"]
      df = df.drop(columns=[f"aspect_{model}", f"sentiment_{model}"])

      print(json.dumps(data))

  df1 = pd.read_csv(f'/dbfs/FileStore/kto/{cfg.domain.name}/predictions.csv', index_col=0)
  df1[f"CASC"] = df1["predicted category"].astype(str)+ " " + df1["predicted polarity"]
  df["CASC"] = df1["CASC"]
  
  df[f"true"] = df["aspect"].astype(str)+ " " + df["sentiment"]

display(df[(df['true'] != df['SBASC']) & (df['true'] != df['SBASCwoSBERT']) & (df['true'] != df['SBASCwoDK']) & (df['true'] != df['SBASCwoDL']) & (df['true'] != df['WBASC']) & (df['true'] != df['CASC'])])
# display(df)

# COMMAND ----------

df1 = pd.read_csv(f'/dbfs/FileStore/kto/{cfg.domain.name}/predictions.csv', index_col=0)
display(df1)
