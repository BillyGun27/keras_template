from base.base_model import BaseModel
import keras
from keras.models import Model
from keras.layers import Input, BatchNormalization, Conv2D, SeparableConv2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D, Dense, Activation, Dropout
from keras import layers
from keras.utils.data_utils import get_file
from keras import optimizers

class XceptionModel(BaseModel):
    def __init__(self, config):
        super(XceptionModel, self).__init__(config)
        self.build_model()

    TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'

    def build_model(self):

        weight_decay=self.config.model.weight_decay
        classes= self.config.model.classes
        dropout=self.config.model.dropout
         
        img_input = Input( shape=(self.config.data_loader.image_size, self.config.data_loader.image_size, 3) )

        x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(img_input)
        x = BatchNormalization(name='block1_conv1_bn')(x)
        x = Activation('relu', name='block1_conv1_act')(x)
        x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
        x = BatchNormalization(name='block1_conv2_bn')(x)
        x = Activation('relu', name='block1_conv2_act')(x)

        residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
        x = BatchNormalization(name='block2_sepconv1_bn')(x)
        x = Activation('relu', name='block2_sepconv2_act')(x)
        x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
        x = BatchNormalization(name='block2_sepconv2_bn')(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
        x = layers.add([x, residual])

        residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = Activation('relu', name='block3_sepconv1_act')(x)
        x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
        x = BatchNormalization(name='block3_sepconv1_bn')(x)
        x = Activation('relu', name='block3_sepconv2_act')(x)
        x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
        x = BatchNormalization(name='block3_sepconv2_bn')(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
        x = layers.add([x, residual])

        residual = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = Activation('relu', name='block4_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
        x = BatchNormalization(name='block4_sepconv1_bn')(x)
        x = Activation('relu', name='block4_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
        x = BatchNormalization(name='block4_sepconv2_bn')(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
        x = layers.add([x, residual])

        for i in range(8):
            residual = x
            prefix = 'block' + str(i + 5)

            x = Activation('relu', name=prefix + '_sepconv1_act')(x)
            x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
            x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
            x = Activation('relu', name=prefix + '_sepconv2_act')(x)
            x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
            x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
            x = Activation('relu', name=prefix + '_sepconv3_act')(x)
            x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
            x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

            x = layers.add([x, residual])

        residual = Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = Activation('relu', name='block13_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
        x = BatchNormalization(name='block13_sepconv1_bn')(x)
        x = Activation('relu', name='block13_sepconv2_act')(x)
        x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
        x = BatchNormalization(name='block13_sepconv2_bn')(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
        x = layers.add([x, residual])

        x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
        x = BatchNormalization(name='block14_sepconv1_bn')(x)
        x = Activation('relu', name='block14_sepconv1_act')(x)

        x = SeparableConv2D(
            2048, (3, 3), padding='same', 
            use_bias=False, name='block14_sepconv2',
            depthwise_regularizer=keras.regularizers.l2(weight_decay),
            pointwise_regularizer=keras.regularizers.l2(weight_decay)
        )(x)
        x = BatchNormalization(name='block14_sepconv2_bn')(x)
        x = Activation('relu', name='block14_sepconv2_act')(x)
        
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        self.model = Model(img_input, x, name='xception')
        
        # load weights
        weights_path = get_file(
            'xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
            self.TF_WEIGHTS_PATH_NO_TOP, cache_subdir='models'
        )
        self.model.load_weights(weights_path)
        
        x = self.model.output
        x = Dropout(dropout)(x)
        logits = Dense(classes, kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
        probabilities = Activation('softmax')(logits)

        self.model = Model(self.model.input, probabilities, name='xception')

        self.model.compile(
              optimizer=optimizers.SGD(lr=1e-2, momentum=0.9, nesterov=True), 
                loss='categorical_crossentropy', metrics=[ 'accuracy', 'top_k_categorical_accuracy']
              )
