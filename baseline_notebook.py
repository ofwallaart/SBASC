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
  cfg = compose("config.yaml", overrides=[f'domain=supermarket', 'model=BERT_baseline', 'environment=databricks', 'ablation=none'])
  print(cfg)
  RUNS = 2
  polarity_list, aspect_list = [], []

  for i in range(RUNS):
      results = BertBaseline(cfg)(cats='polarities')
      polarity_list.append(results)
      
  acc, prec, rec, f1 = 0, 0, 0, 0
  for item in polarity_list:
      acc += item['accuracy']
      prec += item['macro avg']['precision']
      rec += item['macro avg']['recall']
      f1 += item['macro avg']['f1-score']

  print(
      f"accuracy: {acc / len(polarity_list)},\t precision: {prec / len(polarity_list)},\t recall: {rec / len(polarity_list)},\t f1-score: {f1 / len(polarity_list)}")
      
  for i in range(RUNS):
      results = BertBaseline(cfg)(cats='categories')
      aspect_list.append(results)
      
  acc, prec, rec, f1 = 0, 0, 0, 0
  for item in aspect_list:
      acc += item['accuracy']
      prec += item['macro avg']['precision']
      rec += item['macro avg']['recall']
      f1 += item['macro avg']['f1-score']

  print(f"accuracy: {acc/len(aspect_list)},\t precision: {prec/len(aspect_list)},\t recall: {rec/len(aspect_list)},\t f1-score: {f1/len(aspect_list)}")
