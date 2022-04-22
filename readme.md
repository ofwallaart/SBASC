# SBASC
Code for seed Sentence Based Aspect and Sentiment Classification.

All software is written in python (https://www.python.org/) and makes use of the PyTorch framework (https://pytorch.org/).

## Installation Instructions:

### prerequisites:
Please make sure you have the following software/packages installed on your machine:
1. [python3](https://docs.python-guide.org/starting/install3/win/) with your preferred package manager (e.g. [Pip](https://pip.pypa.io/en/stable/))
2. Set up a virtual environment using either [virtualenv](http://pypi.org/project/virtualenv)

or, use anaconda to easily install python (our preferred method):
1. [Download and install Anaconda](https://docs.anaconda.com/anaconda/install/)
2. [Create a new anaconda enviroment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)

We highly recommend using a GPU enabled machine (either locally or remote) for running the models in PyTorch. In our experiments we use a single Tesla V100 GPU (`Standard_NC6s_v3` Azure Virtual Machine). Experience with PyTorch may vary in terms of processing time but when your machine has an NVIDIA GPU it will to harness the full power of PyTorch's CUDA support. For more details on configuration an installation visit [this](https://pytorch.org/get-started/locally/) page.

### Setup Environment:
1. Activate your environment (when using conda run the command `conda activate ENV_NAME`)
2. install the packages listed in requirements.txt. For conda run `conda install --file requirements.txt` from the root directory.

### Download required files:
For English `restaurant` and `laptop` domain data should be placed in `data/restaurant` and `data/laptop` directories respectively. The data can be acquired from [Huang et al.](https://github.com/teapot123/JASen).

For Dutch `restaurant` domain create the `data/restaurant-nl` directory and run the code in `data/dataprep.py`. This will load restaurant data from a large corpus to train the model as well as create an annotated test set from SemEval-2016 data. For this first download the [DU_REST_SB1_TEST.xml](http://metashare.ilsp.gr:8080/repository/browse/semeval-2016-absa-restaurant-reviews-dutch-gold-test-data-phase-b-subtask-1/e514547ccf3811e590e8842b2b6a04d78029b1159e9f4cfc9eed2e7d3df2292c/) from SemEval and place it in the `data/dataprep.py` directory.

### Download or train DK-BERT models:
This code uses BERT (post-)trained models from review corpus to create a Domain Knowledge BERT model. One can choose to train this model using the code from [Xu et al.](https://github.com/howardhsu/BERT-for-RRC-ABSA) by adjusting inputs and running the `transformers/script/run_pt.sh` command.

However for the English `restaurant` and `laptop` domain one can use the pre-trained models from Hugging Face [here](https://huggingface.co/activebus/BERT-DK_rest) and [here](https://huggingface.co/activebus/BERT-DK_laptop) respectively. The model for the Dutch `restaurant` domain can be found [here]() in zipped format.

## Run Software:

### Configuration:
Configuring model settings (most importantly the number of topic classes and their seed sentences) is done in `.yaml` files. The [hydra package](https://hydra.cc/) is used for implementing these configuration files. When training the model one can choose which configuration file to execute. The following configuration files are relevant and contain the following information:

| Composition  | Filename (example)      | Description                                                                                                                                                                                                                                                                                                     |
|--------------|-------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| -            | `config.yaml`           | Starting point of configuration. Contains global variables such as GPU enabling `cuda` and number of epochs to run the model `epochs`. Also contains the default compositions.                                                                                                                                  |
| domain       | `restaurant.yaml`       | Contains domain specific configuration. Most importantly the seed words lists for aspects and sentiment as well as locations to the used BERT models and hyperparameters.                                                                                                                                       |
| model        | `SBASC.yaml`            | Does not contain any configuration at the moment, since there are not many only model specific configurations. They are usually dependent on the datasets used.                                                                                                                                                 |
| domain_model | `restaurant_SBASC.yaml` | Contains configuration for a specific domain&model combination. If this file exists it will overwrite specifications from the domain composition. E.g. in `restaurant_SBASC.yaml` seed sentences are given for each aspect and sentiment instead of seed words, furthermore different hyperparameters are used. |
| environment  | `local.yaml`            | Contains path variables for different environments. When running code in remotely filepaths are different when compared to running the code locally.                                                                                                                                                            |
| ablation     | `woDK.yaml`             | Determines if some parts of the model should be disabled. e.g. for `woDK` the domain knowledge adjusted BERT model is replaced with the standard BERT implementationm (`GroNLP/bert-base-dutch-cased` for Dutch).                                                                                               |

### Run the models:
To run a single run of a model with default configuration, execute the command:
```bash
python run.py
```

Due to usage of Hydra you can override values in the loaded config from the command line:
```bash
python run.py domain=restaurant3 model=CASC ablation=none
```

## Code explanation:
The code contains the following main files that can be run: `run.py`, `run_hyperparameter.py`
- `run.py`: Code to run a single train and test run of the specified model. Configuration can be done in the command line. Results are stored in `results\DOMAIN\MODEL\`
- `run_hyperparameter.py`: Code to run a hyperparameter trial of the specified model. Configuration can be done in the command line. Optimal hyperparameters are stored in `results\MODEL\results.pkl`
- - `run_helpers.py`: Code to run non-essential code, but relevant for result generation. Code contains a function to generate a p-value table comparing different methods using a single sided t-test and a function that graphs different loss function to compare their behaviour.
- `models\BERT_baseline\main.py`: PyTorch implementation of a BERT baseline model. It uses a pre-trained BERT language model (12-layer, 768-hidden, uncased), with a classification layer on top.
- `models\CASC\main.py`: PyTorch implementation of weakly supervised model using post-trained DK-BERT and a small set of seed words for labeled data preparation. Next a neural network using labeled data is used for aspect and sentiment prediction. Code is implemented from [Kumar et al.](https://github.com/Raghu150999/UnsupervisedABSA)
- `models\SBASC\main.py`: PyTorch implementation of seed Sentence Based Aspect and Sentiment Classification model framework using the seed sentence labeler
  - `models\SBASC\labeler_sentence.py`: The seed sentence labeling algorithm for the SBASC model
  - `models\SBASC\trainer.py`: PyTorch helper class to train the neural network.
  - `models\SBASC\model.py`: BERT based neural network model using focal loss for jointly predicting aspect category and sentiment.
- `models\WBASC\main.py`: PyTorch implementation of seed Word Based Aspect and Sentiment Classification model framework using the seed word sentence labeler
  - `models\SBASC\labeler.py`: The seed word labeling algorithm for the WBASC model
- `models\CosSim_baselines\cos_sim.py`: Baseline implementation using a word2vec algorithm using seed words
  - `models\CosSim_baselines\cos_sim_sentence.py`: Baseline implementation using a word2vec algorithm where full seed sentences are averaged and compared

The code contains the following main directories:
- `conf\`: Location for the Hydra configuration `.yaml` files.
- `data\`: Location for the data required by the methods. Each domain should have its own subdirectory. Program generated data will also be stored in the directory.
- `results\`: Contains the stored results for each domain and model stored in subdirectories (e.g. `results\DOMAIN\MODEL\`).

## Related Work: ##
This code uses ideas and code of the following related papers:

- Kumar, A., Gupta, P., Balan, R., Neti, L. B. M., & Malapati, A. (2021). BERT Based Semi-Supervised Hybrid Approach for Aspect and Sentiment Classification. Neural Processing Letters, 53(6), 4207-4224.
- Xu, H., Liu, B., Shu, L., & Yu, P. S. (2019). BERT post-training for review reading comprehension and aspect-based sentiment analysis. arXiv preprint arXiv:1904.02232.