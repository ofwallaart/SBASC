from hydra import compose, initialize

from models.SBASC.labeler_sentence import Labeler
from models.SBASC.trainer import Trainer


class SBASC:
    def __init__(self, cfg):
        self.name = 'SBASC'
        self.cfg = cfg
        self.params = cfg.domain.params

    def __call__(self, load=True):
        labeler = Labeler(self.cfg)
        labeler(load=load)

        trainer = Trainer(self.cfg, **self.params)
        dataset = trainer.load_training_data()
        trainer.train_model(dataset)
        trainer.evaluate()

    def hypertuning(self, params):
        trainer = Trainer(self.cfg, *params)
        dataset = trainer.load_training_data()
        loss, acc = trainer.train_model(dataset, hyper=True)

        return loss, acc


if __name__ == '__main__':
    initialize(config_path="conf")
    config = compose("config.yaml", overrides=['model=SBASC'])
    SBASC(config)()
