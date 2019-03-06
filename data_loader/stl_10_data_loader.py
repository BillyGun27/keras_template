from base.base_data_loader import BaseDataLoader
from keras.preprocessing.image import ImageDataGenerator

def preprocess_input(x):
        x /= 255.0
        x -= 0.5
        x *= 2.0
        return x

class Stl10DataLoader(BaseDataLoader):
    def __init__(self, config):
        super(Stl10DataLoader, self).__init__(config)

        self.data_generator = ImageDataGenerator(
            data_format='channels_last',
            preprocessing_function=preprocess_input
        )

        self.train_generator = self.data_generator.flow_from_directory(
            'data/img/train', 
            target_size=(299, 299),
            batch_size=64
        )

        self.val_generator = self.data_generator.flow_from_directory(
            'data/img/val', shuffle=False,
            target_size=(299, 299),
            batch_size=64
        )

        self.test_generator = self.data_generator.flow_from_directory(
            'data/img/test', shuffle=False,
            target_size=(299, 299),
            batch_size=1
        )


    def get_train_data_generator(self):
        return self.train_generator

    def get_val_data_generator(self):
        return self.val_generator

    def get_test_data_generator(self):
        return self.test_generator
