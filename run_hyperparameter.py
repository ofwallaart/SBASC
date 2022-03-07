import pickle
import time
import os
import json
from bson import json_util
from functools import partial
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

from models.BERT_baseline.main import BertBaseline
from models.SBASC.main import SBASC
from models.WBASC.main import WBASC
from models.SBASC.config import config

modelSpace = {
    'SBASC': [
        hp.choice('learning_rate', [0.00001, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]),
        hp.choice('beta1', [0.8, 0.9, 0.95, 0.97, 0.99]),
        hp.choice('beta2', [0.92, 0.95, 0.97, 0.99, 0.999]),
        hp.choice('batch_size', [12, 24, 36, 48, 60]),
        hp.choice('gamma1', [0, 0.5, 1, 2, 3, 4]),
        hp.choice('gamma2', [0, 0.5, 1, 2, 3, 4]),
    ],
    'WBASC': [
        hp.choice('learning_rate', [0.00001, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]),
        hp.choice('beta1', [0.8, 0.9, 0.95, 0.97, 0.99]),
        hp.choice('beta2', [0.92, 0.95, 0.97, 0.99, 0.999]),
        hp.choice('batch_size', [12]), #24, 36, 48, 60
        hp.choice('gamma1', [0, 0.5, 1, 2, 3, 4]),
        hp.choice('gamma2', [0, 0.5, 1, 2, 3, 4]),
    ],
    'BertBaseline': [
        hp.choice('learning_rate', [0.00001, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]),
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


def save_json_result(file_name, model_name, result):
    """
    Method obtained from Trusca et al. (2020). Save json to a directory and a filename.
    :param file_name:
    :param model_name:
    :param result:
    :return:
    """
    result_name = f'{file_name}.json'
    if not os.path.exists(f"{config['base_path']}results/{config['domain']}/{model_name}/"):
        os.makedirs(f"{config['base_path']}results/{config['domain']}/{model_name}")
    with open(os.path.join(f"{config['base_path']}results/{config['domain']}/{model_name}/", result_name), 'w') as f:
        json.dump(
            result, f,
            default=json_util.default, sort_keys=True,
            indent=4, separators=(',', ': ')
        )



def run_trials(models, max_evals = 30):
    for model in models:
        print(f"Run a trial for: {model.name}")

        print("Attempt to resume a past training if it exists:")
        try:
            # https://github.com/hyperopt/hyperopt/issues/267
            trials = pickle.load(open(f"{config['base_path']}results/{config['domain']}/{model.name}/results.pkl", "rb"))
            print("Found saved Trials! Loading...")
            max_evals = len(trials.trials) + max_evals
            print(f"Rerunning from {trials.trials} trials to add another {max_evals}.")
        except:
            trials = Trials()
            print(f"Starting from scratch: {max_evals} new trials.")

        fmin_objective = partial(objective, model=model)
        best = fmin(fmin_objective, # lambda x: x ** 2,
                    space=modelSpace[model.name], # hp.uniform('x', -10, 10),
                    algo=tpe.suggest,
                    max_evals=max_evals,
                    trials=trials)

        if not os.path.exists(f"{config['base_path']}results/{config['domain']}/{model.name}/"):
            os.makedirs(f"{config['base_path']}results/{config['domain']}/{model.name}")
        pickle.dump(trials, open(f"{config['base_path']}results/{config['domain']}/{model.name}/results.pkl", "wb"))

        print("\nOPTIMIZATION STEP COMPLETE.\n")
        print(best)


if __name__ == '__main__':
    models = [SBASC()]
    run_trials(models, 30)
