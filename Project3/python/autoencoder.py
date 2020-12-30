import sys
import argparse

import numpy as np
import tensorflow.keras

from tensorflow.keras.layers import Dropout,Conv2D,MaxPooling2D,UpSampling2D,Input,Reshape,Dense,Flatten,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.optimizers import RMSprop

from sklearn.model_selection import train_test_split

def readImages(filename):
    with open(filename, mode='rb') as bytestream:
        # throw away magic number
        bytestream.read(4)
        # read the number of images from the metadata
        numOfImages = int.from_bytes(bytestream.read(4), byteorder='big')
        # read the number of rows from the metadata
        numOfRows = int.from_bytes(bytestream.read(4), byteorder='big')
        # read the number of columns from the metadata
        numOfColumns = int.from_bytes(bytestream.read(4), byteorder='big')

        # read actual image data
        buf = bytestream.read(numOfRows * numOfColumns * numOfImages)
        # convert data from bytes to numpy array
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        # reshape so we can access the data of an image
        data = data.reshape(numOfImages, numOfRows, numOfColumns)

        return data


def autoencoder(trainData, trainLabels, valData, valLabels, batch_size, epochs):

    # extracting row and column information
    numberOfRows = trainData.shape[1]
    numberOfColumns = trainData.shape[2]

    # implementing our cnn using the given template from eclass
    # create an empty model
    model = Sequential()

    # adding input layer
    model.add(Input(shape = (numberOfRows, numberOfColumns, 1)))

    # including encoding layers
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # adding flat and dense layer 
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))

    # adding new dense and reshape layer
    model.add(Dense(1152, activation='relu'))
    model.add(Reshape((3,3,128)))

    # inlcuding decoding layers
    model.add(Conv2DTranspose(64, (3, 3), strides=(2,2), activation='relu', padding='VALID'))
    model.add(BatchNormalization())

    model.add(Conv2DTranspose(32, (3, 3), strides=(2,2), activation='relu', padding='same'))
    model.add(BatchNormalization())

    # adding final layer with sigmoid activation function
    model.add(Conv2DTranspose(1, (3, 3), strides=(2,2), activation='sigmoid', padding='same'))

    model.compile(loss = 'mean_squared_error', optimizer = RMSprop())

    model.summary()
    
    model.fit(trainData, trainLabels, 
                        validation_data=(valData, valLabels),
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1)
    
    # finally we have to make sure that history object and model are returned
    return model


def main():

    datasetFilePath = "../originalSpace/trainData"

    # read the data into memory once
    inputData = readImages(datasetFilePath)
    inputData = inputData.reshape(-1, 28,28, 1)
    inputData = inputData / np.max(inputData)


    train_X,valid_X,train_ground,valid_ground = train_test_split(inputData,
                                                                inputData,
                                                                test_size=0.2,
                                                                random_state=13)

    
    batch_size = 64
    epochs = 10

    model = autoencoder(train_X, train_ground, valid_X, valid_ground, batch_size, epochs)

    # saving the model
    model.save("../models/autoencoderTest.h5")

if __name__ == "__main__":
    main()