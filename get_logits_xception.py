from data_loader.stl_10_logits_loader import Stl10LogitsLoader
#from models.xception_model import XceptionModel
from keras.models import Model,load_model
from trainers.generator_trainer import GeneratorModelTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from tqdm import tqdm

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
    data_loader = Stl10LogitsLoader(config)

    print('Create the model.')
    #model = XceptionModel(config)

    #model.load("datasets/model_data/xception.h5")
    model = load_model("datasets/model_data/xception.h5")
    model.layers.pop()
    model = Model(model.input, model.layers[-1].output)

    print('Get Logits.')
    batches = 0
    train_logits = {}

    for x_batch, y_batch, name_batch in tqdm(data_loader.train_generator):
        
        batch_logits = model.predict_on_batch(x_batch)
        
        for i, n in enumerate(name_batch):
            #print(n)
            train_logits[n] = softmax(batch_logits[i])
        
        batches += 1
        if batches >= 5000//50: # 5000/64
            break
    
    np.save('train_logits.npy', train_logits)
    print(train_logits[0])
  
    batches = 0
    test_logits = {}
    numb = 0
    for x_batch, _, name_batch in tqdm(data_loader.test_generator ):
        
        batch_logits = model.predict_on_batch(x_batch)

        for i, n in enumerate(name_batch):
            test_logits[n] = softmax(batch_logits[i])
        
        batches += 1
        if batches >= 3000//50: # 3000/64
            break

    np.save('test_logits.npy', test_logits)
    print(test_logits[0])

 #   print('Create the trainer')
 #   trainer = GeneratorModelTrainer(model.model, data_loader.train_generator, data_loader.val_generator , config)

 #   print('Start training the model.')
 #   trainer.train()

# Define a manual softmax function
def softmax(x):
    return np.exp(x)/(np.exp(x).sum())

if __name__ == '__main__':
    main()
