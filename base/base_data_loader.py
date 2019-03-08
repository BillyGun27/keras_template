class BaseDataLoader(object):
    def __init__(self, config):
        self.config = config

    def get_train_data_generator(self):
        raise NotImplementedError

    def get_test_data_generator(self):
        raise NotImplementedError

    def get_val_data_generator(self):
        raise NotImplementedError
