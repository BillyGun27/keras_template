from data_loader.stl_10_distillation_loader import Stl10DistillationLoader
from models.squeezenet_model import SqueezenetModel
from trainers.generator_trainer import GeneratorModelTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    print('Create the data generator.')
    data_loader = Stl10DistillationLoader(config)

    print('Create the model.')
    model = SqueezenetModel(config)

    print('Create the trainer')
    trainer = GeneratorModelTrainer(model.model, data_loader.train_generator, data_loader.val_generator , config)

    print('Start training the model.')
    trainer.train()


if __name__ == '__main__':
    main()
