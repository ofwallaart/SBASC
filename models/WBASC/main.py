from models.WBASC.labeler import Labeler
from models.SBASC.trainer import Trainer
from models.WBASC.config import *


class WBASC:
    def __init__(self):
        self.name = 'WBASC'

    def __call__(self):
        labeler = Labeler()
        labeler(load=True)

        trainer = Trainer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, batch_size=batch_size, gamma1=gamma1, gamma2=gamma2)
        dataset = trainer.load_training_data()
        trainer.train_model(dataset)
        trainer.evaluate()

    def hypertuning(self, params):
        trainer = Trainer(*params)
        dataset = trainer.load_training_data()
        loss, acc = trainer.train_model(dataset, hyper=True)

        return loss, acc


if __name__ == '__main__':
    WBASC()()
