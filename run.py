import pandas as pd
from hydra import compose, initialize
import os
import json
import time

from models.SBASC.main import SBASC
from models.WBASC.main import WBASC


def store_run_results(results, path, model_name, ablation):
    date = time.strftime("%Y%m%d_%H%M%S")

    if not os.path.exists(f"{path}/{model_name}/"):
        os.makedirs(f"{path}/{model_name}")

    json_object = json.dumps({
        'polarity': results[0],
        'aspect': results[1],
        'predicted_pol': results[2],
        'predicted_asp': results[3],
    })
    with open(f"{path}/{model_name}/run_{date}_{ablation}.json", "w") as outfile:
        outfile.write(json_object)

    df_output_path = f"{path}/{model_name}/results.csv"

    d = {
      'timestamp': [date, date],
      'type': ['polarity', 'aspect'],
      'ablation': [ablation, ablation],
      'accuracy': [results[0]['accuracy'], results[1]['accuracy']],
      'precision': [results[0]['macro avg']['precision'], results[1]['macro avg']['precision']],
      'recall': [results[0]['macro avg']['recall'], results[1]['macro avg']['recall']],
      'f1-score': [results[0]['macro avg']['f1-score'], results[1]['macro avg']['f1-score']]
    }

    df_new = pd.DataFrame(data=d)
    df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], format='%Y%m%d_%H%M%S')

    if os.path.exists(df_output_path):
      df_old = pd.read_csv(df_output_path, index_col=0)
      pd.concat([df_old, df_new]).reset_index(drop=True).to_csv(df_output_path, mode='w')
    else:
      df_new.to_csv(df_output_path, mode='w')


if __name__ == "__main__":
    with initialize(config_path="conf"):
        cfg = compose("config.yaml", overrides=['domain=restaurant3', 'model=WBASC', 'ablation=woSBERT'])
        # Run SBASC
        results = WBASC(cfg).labeler(load=False)

        store_run_results(results, cfg.result_path_mapper, cfg.model.name, cfg.ablation.name)
