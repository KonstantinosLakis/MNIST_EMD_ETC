import sys
import argparse
import math

import numpy as np
import keras

from tensorflow.keras.layers import Dropout,Conv2D,MaxPooling2D,UpSampling2D,Input,Reshape,Dense,Flatten,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential,load_model
from keras.optimizers import RMSprop

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

import pandas as pd

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

        return data, numOfImages

def writeImages(filename, normal_predictions, numOfImages, numOfRows, numOfColumns):
    with open(filename, mode='wb') as binary_file:
        # write magic number
        binary_file.write((69).to_bytes(4, byteorder='big'))
        # write number of images
        binary_file.write((numOfImages).to_bytes(4, byteorder='big'))
        # write number of rows
        binary_file.write((numOfRows).to_bytes(4, byteorder='big'))
        # write number of columns
        binary_file.write((numOfColumns).to_bytes(4, byteorder='big'))

        # write pixels
        for image in normal_predictions:
            for pixel in image:
                binary_file.write((int(pixel)).to_bytes(1, byteorder='big')) 


def buildCompleteModel():

    savedModelPath = "../models/autoencoder.h5"

    # we need to load the saved model and add the encoder layers to a new model
    savedModel = load_model(savedModelPath)

    # create a new model and insert the layers
    fullModel = Sequential()

    # loop through all encoder layers which are floor(layers / 2) + 2
    numberOfEncoderLayers = math.floor(len(savedModel.layers) / 2) + 2

    for layerNumber in range(numberOfEncoderLayers):
        fullModel.add(savedModel.layers[layerNumber])
    
    fullModel.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

    return fullModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d')
    parser.add_argument('-q')
    parser.add_argument('-od')
    parser.add_argument('-oq')
    args = parser.parse_args()

    datasetFilePath = args.d
    querysetFilePath = args.q

    outDatasetFilePath = args.od
    outQuerysetFilePath = args.oq

    # read the data into memory once
    dataset, dNumOfImages = readImages(datasetFilePath)
    dataset = dataset.reshape(-1, 28,28, 1)
    dataset = dataset / np.max(dataset)

    queryset, qNumOfImages = readImages(querysetFilePath)
    queryset = queryset.reshape(-1, 28,28, 1)
    queryset = queryset / np.max(queryset)

    # compute predictions
    model = buildCompleteModel()
    predictions = model.predict(dataset)

    # normalize predictions
    predictions = predictions / np.max(predictions)
    predictions = predictions * 255
    predictions = predictions.astype(int)

    # write new data
    writeImages(outDatasetFilePath, predictions, dNumOfImages, 1, len(predictions[0]))
    writeImages(outQuerysetFilePath, predictions, qNumOfImages, 1, len(predictions[0]))



if __name__ == "__main__":
    main()