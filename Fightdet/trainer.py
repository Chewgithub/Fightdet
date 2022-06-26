import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger


from Fightdet.model import model_instantiate
from Fightdet.data_generator import instantiate_generator


# Define callbacks
def early_stopping(es_patience):
    '''
    Read a patience value for early stopping criteria, return EarlyStopping class object

    Parameters:
        es_patience: int
    Returns:
        es: EarlyStopping class object
    '''
    # early stopping
    es = EarlyStopping(patience=es_patience, restore_best_weights=True,monitor='val_loss')
    return es

# Define CSVLogger
def csv_logger(traininglog_filename):
    '''
    Read a file path for early stopping criteria, return EarlyStopping class object

    Parameters:
        es_patience: int
    Returns:
        es: EarlyStopping class object
    '''
    # log epoch results in a CSV file
    csv_log = CSVLogger(traininglog_filename)
    return csv_log

def model_compilation():
    '''
    Compile a instantiated model with optimizer:Adam, Loss:binary_cross_entropy,
    metrics: precision, recall, accuracy

    Parameters:
        None
    Returns:
        compiled model
    '''
    #adam optimizer
    adam = Adam(
    learning_rate=0.001,
    beta_1=0.95,
    beta_2=0.999,
    epsilon=1e-01,
    amsgrad=False)

    #instantiate a model
    cnn_model=model_instantiate()

    #compile model
    cnn_model.compile(optimizer=adam,
                 loss='binary_crossentropy',
                 metrics=[Precision(name='precision'), Recall(name='recall'), 'accuracy'])
    return cnn_model

def model_fitting(cnn_model,train_generator,val_generator,es,csv_logger):
    '''
    Fit a compiled model with train and val data generator. This function also takes
    in early stopping and csv_logger class from tensorflow.keras.callbacks

    Parameters:
        cnn_model(compiled_model)
        train_generator(Datagenerator)
        val_generator(Datagenerator)
        es(EarlyStoppping Class)
        csv(CSVLogger Class)
    Returns:
        compiled and fitted model
    '''
    num_epochs  = 100
    num_workers = 16
    max_queue = 5

    #fitting the model
    history_base = cnn_model.fit(x=train_generator,
                            verbose=1,
                            validation_data=val_generator,
                            callbacks=[es, csv_logger],
                            epochs=num_epochs,
                            workers=num_workers,
                            max_queue_size=max_queue,
                            steps_per_epoch=len(train_generator),
                            validation_steps=len(val_generator)
                            )
    return cnn_model

if __name__=="__main__":

    #saving training log as csv,  replace file path to your desired path
    traininglog_filename=os.path.join(os.getcwd(),'training_log','model_training_log.csv')
    batch_size=5
    es_patience=5

    '''
    instantiate data generator, replace file path to folder that contain all
    all raw .npy file
    '''
    npy_dataset_folder = 'raw_data/npy_raw_data'
    train_generator, val_generator=instantiate_generator(npy_dataset_folder, batch_size)

    #model compilation
    cnn_model=model_compilation()
    es=early_stopping(es_patience)
    csv_log=csv_logger(traininglog_filename)

    #model fitting
    cnn_model=model_fitting(cnn_model,train_generator,val_generator,es,csv_log)

    #saving model, replace the file paht to your desired path
    save_model_path=os.path.join(os.getcwd(),'model_collection','cnn_model')
    cnn_model.save(save_model_path)
