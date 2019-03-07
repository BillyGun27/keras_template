from base.base_model import BaseModel
import keras
from keras.applications.mobilenetv2 import MobileNetV2
from keras.models import Model
from keras.layers import Activation, GlobalAveragePooling2D, Dropout, Dense, Input
from keras import layers
from keras.utils.data_utils import get_file
from keras import optimizers

class MobilenetModel(BaseModel):
    def __init__(self, config):
        super(MobilenetModel, self).__init__(config)
        self.build_model()

    def build_model(self):

        weight_decay=self.config.model.weight_decay
        classes= self.config.model.classes
        alpha=self.config.model.alpha,  
        dropout=self.config.model.dropout

        input_shape = (self.config.data_loader.image_size, self.config.data_loader.image_size, 3) 
    
        base_model = MobileNetV2(
        include_top=False, weights='imagenet', 
        input_shape=input_shape, alpha=alpha
        )
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(dropout)(x)
        logits = Dense(10, kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
        probabilities = Activation('softmax')(logits)
       
        self.model = Model(base_model.input, probabilities , name='mobilenet')
        

        self.model.compile(
              optimizer=optimizers.SGD(lr=1e-2, momentum=0.9, nesterov=True), 
                loss='categorical_crossentropy', metrics=[ 'accuracy', 'top_k_categorical_accuracy']
              )
