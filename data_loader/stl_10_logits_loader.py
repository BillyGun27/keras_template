from base.base_data_loader import BaseDataLoader
from utils.image_preprocessing_logits import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input


class Stl10LogitsLoader(BaseDataLoader):
    def __init__(self, config):
        super(Stl10LogitsLoader, self).__init__(config)

        self.data_generator = ImageDataGenerator(
            data_format='channels_last',
            preprocessing_function=preprocess_input
        )

        self.train_generator = self.data_generator.flow_from_directory(
            'datasets/img/train', 
            target_size=(self.config.data_loader.image_size , self.config.data_loader.image_size ),
            batch_size=self.config.trainer.batch_size
        )

        self.test_generator = self.data_generator.flow_from_directory(
            'datasets/img/test', shuffle=False,
            target_size=(self.config.data_loader.image_size , self.config.data_loader.image_size),
            batch_size=self.config.trainer.batch_size
        )



    def get_train_data_generator(self):
        return self.train_generator

    def get_test_data_generator(self):
        return self.test_generator

