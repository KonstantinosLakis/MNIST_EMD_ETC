import sys
import argparse

import math
import numpy as np
import keras
from matplotlib import pyplot as plt

from keras.layers import Dropout,Conv2D,MaxPooling2D,UpSampling2D,Input,Flatten,Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.optimizers import RMSprop
from keras.utils import to_categorical


from sklearn.model_selection import train_test_split
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

        return data
    
  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d')
    parser.add_argument('-model')
    parser.add_argument('-k')
    parser.add_argument('-o')
    args = parser.parse_args()

    datasetFilePath = args.d
    modelFilePath = args.model
    numberOfClusters = int(args.k)
    outputFilePath = args.o

    # read the data into memory once
    inputData = readImages(datasetFilePath)
    inputData = inputData.reshape(-1, 28,28, 1)
    inputData = inputData / np.max(inputData)

    # read the model into memory so we can classify and then create the cluster file
    classifierModel = keras.models.load_model(modelFilePath)

    # predict the classes of the input data
    classes = [np.argmax(oneHot) for oneHot in classifierModel.predict(inputData)]

    # now write cluster output based on classes
    clusters = [[] for _ in range(numberOfClusters)]

    for predictionIndex in range(len(classes)):
        clusters[classes[predictionIndex]].append(predictionIndex)

    with open(outputFilePath, mode='w') as file:
        clusterIndex = 1
        for cluster in clusters:
            file.write("CLUSTER-" + str(clusterIndex) + " { size: " + str(len(cluster)))
            for imageNumber in cluster:
                file.write(", " + str(imageNumber))
            file.write("}\n")
            clusterIndex += 1
   
if __name__ == "__main__":
    main()
