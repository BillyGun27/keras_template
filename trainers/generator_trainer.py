from base.base_trainer import BaseTrain
import os
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

class GeneratorModelTrainer(BaseTrain):
    def __init__(self, model, train_gen ,val_gen , config):
        super( GeneratorModelTrainer , self).__init__(model, train_gen ,val_gen , config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.h5' % self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )
        self.callbacks.append(
            EarlyStopping(monitor='val_acc', patience=4, min_delta=0.01)
        )
        self.callbacks.append(
            ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=2, epsilon=0.007)
        )
    '''
        if hasattr(self.config,"comet_api_key"):
            from comet_ml import Experiment
            experiment = Experiment(api_key=self.config.comet_api_key, project_name=self.config.exp_name)
            experiment.disable_mp()
            experiment.log_multiple_params(self.config)
            self.callbacks.append(experiment.get_keras_callback())
    '''
    def train(self):
        history = self.model.fit_generator(
            self.train_gen, 
            steps_per_epoch = self.train_gen.n//self.train_gen.batch_size,
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            #batch_size=self.config.trainer.batch_size,
            callbacks=self.callbacks,
            validation_data=self.val_gen, validation_steps=self.val_gen.n//self.val_gen.batch_size, 
            workers=4
        )
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])
