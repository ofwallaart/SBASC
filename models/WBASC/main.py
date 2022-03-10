from hydra import compose, initialize

from models.WBASC.labeler import Labeler
from models.SBASC.trainer import Trainer
from models.WBASC.config import *


class WBASC:
    def __init__(self, cfg):
        self.name = 'WBASC'
        self.cfg = cfg
        self.params = cfg.domain.params

    def __call__(self, load=True, evaluate=True):
        labeler = Labeler(self.cfg)
        trainer = Trainer(self.cfg, **self.params)

        if self.cfg.ablation.name == 'WithoutDomainKnowledge':
            results = labeler(load=load, evaluate=True)
        else:
            labeler(load=load, evaluate=evaluate)
            dataset = trainer.load_training_data()
            trainer.train_model(dataset)
            results = trainer.evaluate()

        return results
    
    def labeler(self, load=True):
        labeler = Labeler(self.cfg)
        labeler(load=load)

    def hypertuning(self, params):
        trainer = Trainer(self.cfg, *params)
        dataset = trainer.load_training_data()
        loss, acc = trainer.train_model(dataset, hyper=True)

        return loss, acc


if __name__ == '__main__':
    initialize(config_path="conf")
    config = compose("config.yaml", overrides=['model=WBASC'])
    WBASC(config)()
