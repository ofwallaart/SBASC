# Databricks notebook source
from pyspark.sql.functions import avg, col, lit, when

df = spark.read.format('csv').options(header='true', inferSchema='true').load('dbfs:/FileStore/kto/results/restaurant-5/WBASC/results.csv')

display(df.groupBy('type', 'ablation').agg(avg('accuracy'), avg('precision'), avg('recall'), avg('f1-score')).sort("ablation","type")) 


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
