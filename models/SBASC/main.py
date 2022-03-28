from hydra import compose, initialize

from models.SBASC.labeler_sentence import Labeler
from models.SBASC.trainer import Trainer


class SBASC:
    def __init__(self, cfg):
        self.name = 'SBASC'
        self.cfg = cfg
        self.params = cfg.domain.params
        self.trainer = None

    def __call__(self, load=True, evaluate=True):
        labeler = Labeler(self.cfg)
        self.trainer = Trainer(self.cfg, **self.params)

        if self.cfg.ablation.name == 'WithoutDeepLearning':
            results = labeler(load=load, evaluate=True)
        else:
            labeler(load=load, evaluate=evaluate)
            dataset = self.trainer.load_training_data()
            self.trainer.train_model(dataset)
            results = self.trainer.evaluate()

        return results
      
    def save(self, name):
        self.trainer.save_model(name)

    def load(self, name):
        self.trainer.load_model(name)

    def labeler(self, load=True):
        labeler = Labeler(self.cfg)
        return labeler(load=load)

    def hypertuning(self, params):
        trainer = Trainer(self.cfg, *params)
        dataset = trainer.load_training_data()
        loss, acc = trainer.train_model(dataset, hyper=True)

        return loss, acc
      
    def predict(self, sentences):
        return self.trainer.predict(sentences)

if __name__ == '__main__':
    initialize(config_path="conf")
    config = compose("config.yaml", overrides=['model=SBASC'])
    SBASC(config)()
