class BaseTrain(object):
    def __init__(self, model, train_gen ,val_gen  , config):
        self.model = model
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.config = config

    def train(self):
        raise NotImplementedError
