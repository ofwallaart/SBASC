# Script to run hyperparameter optimization for the models.

import pickle
import time
import os
import json
from bson import json_util
from functools import partial
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import hydra
from omegaconf import DictConfig

from models.BERT_baseline.main import BertBaseline
from models.SBASC.main import SBASC
from models.WBASC.main import WBASC
from models.CASC.main import CASC

# Define hyperparameter spaces for each model
modelSpace = {
    'SBASC': [
        hp.choice('learning_rate', [1e-6, 1e-5, 1e-4, 5e-3, 1e-3, 5e-2, 1e-2, 0.02, 0.05, 0.1]),
        hp.choice('beta1', [0.8, 0.9, 0.95, 0.97, 0.99]),
        hp.choice('beta2', [0.92, 0.95, 0.97, 0.99, 0.999]),
        hp.choice('batch_size', [12, 24, 36, 48, 60]),
        hp.choice('gamma1', [0, 0.5, 1, 2, 3, 4]),
        hp.choice('gamma2', [0, 0.5, 1, 2, 3, 4]),
    ],
    'WBASC': [
        hp.choice('learning_rate', [1e-6, 1e-5, 1e-4, 5e-3, 1e-3, 5e-2, 1e-2, 0.02, 0.05, 0.1]),
        hp.choice('beta1', [0.8, 0.9, 0.95, 0.97, 0.99]),
        hp.choice('beta2', [0.92, 0.95, 0.97, 0.99, 0.999]),
        hp.choice('batch_size', [12, 24, 36, 48, 60]),
        hp.choice('gamma1', [0, 0.5, 1, 2, 3, 4]),
        hp.choice('gamma2', [0, 0.5, 1, 2, 3, 4]),
    ],
    'CASC': [
        hp.choice('learning_rate', [1e-6, 1e-5, 1e-4, 5e-3, 1e-3, 5e-2, 1e-2, 0.02, 0.05, 0.1]),
        hp.choice('beta1', [0.8, 0.9, 0.95, 0.97, 0.99]),
        hp.choice('beta2', [0.92, 0.95, 0.97, 0.99, 0.999]),
        hp.choice('batch_size', [12, 24, 36, 48, 60]),
    ],
    'BertBaseline': [
        hp.choice('learning_rate', [1e-6, 1e-5, 1e-4, 5e-3, 1e-3, 5e-2, 1e-2, 0.02, 0.05, 0.1]),
        hp.choice('beta1', [0.8, 0.9, 0.95, 0.97, 0.99, 0.99]),
        hp.choice('beta2', [0.92, 0.95, 0.97, 0.99, 0.999]),
        hp.choice('batch_size', [12, 24, 36, 48, 60]),
    ]
}


def objective(params, model):
    print(params)
    min_loss, acc = model.hypertuning(params)

    result = {
        'loss': min_loss,
        'status': STATUS_OK,
        'space': params,
        'accuracy': acc,
        'eval_time': time.time()
    }

    # save_json_result('acc_' + str(acc) + '_loss_' + str(min_loss), str(model.name), result)

    return result


def save_json_result(file_name, model_name, result, result_path):
    """
    Method obtained from Trusca et al. (2020). Save json to a directory and a filename.
    :param file_name:
    :param model_name:
    :param result:
    :return:
    """
    result_name = f'{file_name}.json'
    if not os.path.exists(f"{result_path}/{model_name}/"):
        os.makedirs(f"{result_path}/{model_name}")
    with open(os.path.join(f"{result_path}/{model_name}/", result_name), 'w') as f:
        json.dump(
            result, f,
            default=json_util.default, sort_keys=True,
            indent=4, separators=(',', ': ')
        )



def run_trials(models, cfg, max_evals = 30):
    """
    Function to run a number of hyperparameter trials
    :param models: list of models we want to perform hyperparameter tuning on
    :param cfg: configuration file
    :param max_evals: maximum number of new evaluation trials we want to perform
    """
    for model in models:
        print(f"Run a trial for: {model.name}")

        # Resume training if we already did some optimization before
        print("Attempt to resume a past training if it exists:")
        try:
            # https://github.com/hyperopt/hyperopt/issues/267
            trials = pickle.load(open(f"{cfg.base_path}results/{cfg.domain.name}/{model.name}/results.pkl", "rb"))
            print("Found saved Trials! Loading...")
            max_evals = len(trials.trials) + max_evals
            print(f"Rerunning from {trials.trials} trials to add another {max_evals}.")
        except:
            trials = Trials()
            print(f"Starting from scratch: {max_evals} new trials.")

        fmin_objective = partial(objective, model=model)
        best = fmin(fmin_objective,
                    space=modelSpace[model.name],
                    algo=tpe.suggest,
                    max_evals=max_evals,
                    trials=trials)

        # Store results
        if not os.path.exists(f"{cfg.result_path_mapper}/{model.name}/"):
            os.makedirs(f"{cfg.result_path_mapper}/{model.name}")
        pickle.dump(trials, open(f"{cfg.result_path_mapper}/{model.name}/results.pkl", "wb"))

        print("\nOPTIMIZATION STEP COMPLETE.\n")
        print(best)


# default: domain=restaurant3 model=SBASC ablation=None]
@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    # Run SBASC/CASC/WBASC
    if cfg.model.name == 'CASC':
        models = [CASC(cfg)]
    elif cfg.model.name == 'WBASC':
        models = [WBASC(cfg)]
    else:
        models = [SBASC(cfg)]
    run_trials(models, cfg, 50)


if __name__ == '__main__':
    my_app()
