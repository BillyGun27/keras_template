from base.base_model import BaseModel
import keras
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, GlobalAveragePooling2D, concatenate
from keras import layers
from keras.utils.data_utils import get_file
from keras import optimizers

class SqueezenetModel(BaseModel):
    def __init__(self, config):
        super(SqueezenetModel, self).__init__(config)
        self.build_model()

    # a building block of the SqueezeNet architecture
    def fire_module(self, number, x, squeeze, expand, weight_decay=None, trainable=False):
        
        module_name = 'fire' + number
        
        if trainable and weight_decay is not None:
            kernel_regularizer = keras.regularizers.l2(weight_decay) 
        else:
            kernel_regularizer = None
        
        x = Convolution2D(
            squeeze, (1, 1), 
            name=module_name + '/' + 'squeeze',
            trainable=trainable, 
            kernel_regularizer=kernel_regularizer
        )(x)
        x = Activation('relu')(x)

        a = Convolution2D(
            expand, (1, 1),
            name=module_name + '/' + 'expand1x1',
            trainable=trainable, 
            kernel_regularizer=kernel_regularizer
        )(x)
        a = Activation('relu')(a)

        b = Convolution2D(
            expand, (3, 3), padding='same',
            name=module_name + '/' + 'expand3x3',
            trainable=trainable, 
            kernel_regularizer=kernel_regularizer
        )(x)
        b = Activation('relu')(b)

        return concatenate([a, b])
    
    def build_model(self):

        weight_decay=self.config.model.weight_decay
        classes= self.config.model.classes
         
        img_input = Input( shape=(self.config.data_loader.image_size, self.config.data_loader.image_size, 3) )

        x = Convolution2D(
        64, (3, 3), strides=(2, 2), name='conv1', 
        trainable=False
        )(img_input) # 111, 111, 64
        
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x) # 55, 55, 64

        x = self.fire_module('2', x, squeeze=16, expand=64) # 55, 55, 128
        x = self.fire_module('3', x, squeeze=16, expand=64) # 55, 55, 128
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x) # 27, 27, 128

        x = self.fire_module('4', x, squeeze=32, expand=128) # 27, 27, 256
        x = self.fire_module('5', x, squeeze=32, expand=128) # 27, 27, 256
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x) # 13, 13, 256
        
        x = self.fire_module('6', x, squeeze=48, expand=192) # 13, 13, 384
        x = self.fire_module('7', x, squeeze=48, expand=192) # 13, 13, 384
        x = self.fire_module('8', x, squeeze=64, expand=256) # 13, 13, 512
        x = self.fire_module('9', x, squeeze=64, expand=256) # 13, 13, 512
        
        x = Dropout(0.5)(x)
        x = Convolution2D(
            10, (1, 1), name='conv10',
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
            kernel_regularizer=keras.regularizers.l2(weight_decay)
        )(x) # 13, 13, 256
        
        x = Activation('relu')(x)
        logits = GlobalAveragePooling2D()(x) # 256#10
        probabilities = Activation('softmax')(logits)
        
        self.model = Model(img_input, probabilities)
        #self.model.load_weights('squeezenet_weights.hdf5', by_name=True)

        self.model.compile(
              optimizer=optimizers.SGD(lr=1e-2, momentum=0.9, nesterov=True), 
                loss='categorical_crossentropy', metrics=[ 'accuracy', 'top_k_categorical_accuracy']
              )
