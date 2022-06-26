#Importing dependencies
import os
import numpy as np

from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    """Data Generator inherited from tensorflow.keras.utils.Sequence
    Args:
        directory(str): the path of dataset, and each sub-folder will be assigned to one class
        batch_size(int): the number of data points in each batch. Loss function and parameter
                    will be updated during training per this batch size.
        shuffle(bool): whether to shuffle the data per epoch
        data augmentation(bool): whether to random flip the video frame (used for trainset)
    """
    def __init__(self, directory, batch_size=1, shuffle=True, data_augmentation=True):
        # Initialize the params
        self.batch_size = batch_size
        self.directory = directory
        self.shuffle = shuffle
        self.data_aug = data_augmentation

        # Load all the save_path of files, and create a dictionary that save the pair of "data:label"
        self.X_path, self.Y_dict = self.search_data()

        # Print basic statistics information, include example of input path and target value(1 or 0)
        print(self.X_path[0],self.Y_dict[self.X_path[0]])
        print(self.X_path[-1],self.Y_dict[self.X_path[-1]])
        self.print_stats()
        return None

    def search_data(self):
        X_path = []
        Y_dict = {}
        # list all kinds of sub-folders
        self.dirs = sorted(os.listdir(self.directory), reverse=True)
        # prepare [0] for NonFight and [1] for Fight
        labels = list(range(len(self.dirs)))
        for i,folder in enumerate(self.dirs):
            folder_path = os.path.join(self.directory,folder)
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path,file)
                # append the each file path, and keep its label
                X_path.append(file_path)
                # add all file path as key and target as value in a dictonary
                Y_dict[file_path] = labels[i]
        return X_path, Y_dict

    def print_stats(self):
        # calculate basic information
        self.n_files = len(self.X_path)
        self.n_classes = len(self.dirs)
        self.indexes = np.arange(len(self.X_path))
        np.random.shuffle(self.indexes)
        # Output states
        print("Found {} files belonging to {} classes.".format(self.n_files,self.n_classes))
        for i,label in enumerate(self.dirs):
            print('%10s : '%(label),i)
        return None

    def __len__(self):
        # calculate the iterations of each epoch
        steps_per_epoch = np.ceil(len(self.X_path) / float(self.batch_size))
        return int(steps_per_epoch)

    def __getitem__(self, index):
        """
        Get the data of each batch
        """
        # get the indexs of each batch
        batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # using batch_indexs to get path of current batch
        batch_path = [self.X_path[k] for k in batch_indexs]
        # get batch data
        batch_x, batch_y = self.data_generation(batch_path)
        return batch_x, batch_y

    def on_epoch_end(self):
        # shuffle the data at each end of epoch(if shuffle=True)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_path):
        # load data into memory
        batch_x = [self.load_data(x) for x in batch_path]
        batch_y = [self.Y_dict[x] for x in batch_path]
        # transfer the data format

        # shape of batch x = (batch_size,64,224,224,3)
        batch_x = np.array(batch_x)
        # shape of batch y = (batch_size)
        batch_y = np.array(batch_y)
        return batch_x, batch_y

    def standardize(self, data):
        mean = np.mean(data)
        std = np.std(data)
        return (data-mean) / std

    def random_flip(self, video, prob):
        s = np.random.rand()
        if s < prob:
          #flipping frame in a video horizontally
          video = np.flip(m=video, axis=2)
        return video

    def load_data(self, path):
        # load the processed .npy files which have 3 channels (1 for gray_frame, 4-5 for optical flows)
        data = np.load(path, mmap_mode='r')
        data = np.float32(data)
        # whether to utilize the data augmentation
        if self.data_aug:
            data = self.random_flip(data, prob=0.5)
        # standardize gray images and optical flows, respectively
        data[...,:1] = self.standardize(data[...,:1])
        data[...,1:] = self.standardize(data[...,1:])
        return data

# Instatiate Generator
def instantiate_generator(batch_size=5):
    '''
    This function instantiate the data generators for train and validation set for CNN model training
    Parameters:
        batch size(int): batch size for model training, default is 5, take your
        processing memory into account if you wish to change it. Model might not fit
        if the system memory is insufficient.

    Returns:
        train_generator(Datagenerator): data generator class for train set
        val_generator(Datagenerator): data generator class for validation set

    '''
    # change the folder path to raw_data folder which contain train and test set
    npy_dataset_folder = 'raw_data/npy_raw_data'
    train_generator = DataGenerator(directory=f'{npy_dataset_folder}/train/',
                                batch_size=batch_size,
                                data_augmentation=True)

    val_generator = DataGenerator(directory=f'{npy_dataset_folder}/test/',
                                batch_size=batch_size,
                                data_augmentation=False)

    return train_generator, val_generator
