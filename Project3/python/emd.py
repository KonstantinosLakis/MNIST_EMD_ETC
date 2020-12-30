import argparse
import bisect
import numpy as np
import math
from scipy.optimize import linprog
from functools import partial

import warnings
warnings.filterwarnings("ignore")


def readLabels(filename):
    with open(filename, mode='rb') as bytestream:
        # throw away magic number
        bytestream.read(4)
        # read the number of labels from the metadata
        numOfLabels = int.from_bytes(bytestream.read(4), byteorder='big')
      
        # read actual label data
        buf = bytestream.read(numOfLabels)
        # convert data from bytes to numpy array
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

        return labels

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
        data = data.reshape(numOfImages, numOfRows * numOfColumns)

        return data

def kNearestNeighbors(queryImage, otherImages, k, distFunc):
    neighborsWithDistances = []
    currId = 0

    for otherImage in otherImages:
        currDistance = distFunc(queryImage, otherImage)

        # if we are still looking for the first k neighbors, simply insert candidate
        if len(neighborsWithDistances) < k:
            bisect.insort(neighborsWithDistances, (currDistance, currId))
        else:
            # else we need to actually compare with current worst
            # and remove if necessary
            if currDistance < neighborsWithDistances[k - 1][0]:
                neighborsWithDistances.pop()
                bisect.insort(neighborsWithDistances, (currDistance, currId))
        

        currId += 1

    return [neighborId for (_, neighborId) in neighborsWithDistances]

def windowIndexToCentroid(width, height, index):
    verticalSlots = 28 // height
    horizontalSlots = 28 // width

    xIndex = index % horizontalSlots
    yIndex = index // horizontalSlots

    centroidX = (xIndex + 0.5) * width
    centroidY = (verticalSlots - yIndex - 0.5) * height

    return (centroidX, centroidY)


def windowDistance(width, height, window1, window2):
    centroid1 = windowIndexToCentroid(width, height, window1)
    centroid2 = windowIndexToCentroid(width, height, window2)

    return math.sqrt(pow(centroid1[0] - centroid2[0], 2) + pow(centroid1[1] - centroid2[1], 2))

def manhattanDistance(image, otherImage):
    s = 0
    for index in range(image.size):
        s += abs(image[index] - otherImage[index])
    return s


def earthMoverDistance(inputWidth, inputHeight, image, otherImage):
    # window dimensions for emd
    windowWidth = inputWidth
    windowHeight = inputHeight

    windowXIndexes = [x for x in range(28 // windowWidth)]
    windowYIndexes = [y for y in range(28 // windowHeight)]

    windowIndex = 0
    imageSignatures = []
    otherImageSignatures = []

    # for every window
    for windowY in windowYIndexes:
        for windowX in windowXIndexes:
            # calculate the actual signatures for current window
            imageSignature = (windowIndex, calculateBrightness(windowWidth, windowHeight, windowX, windowY, image))
            otherImageSignature = (windowIndex, calculateBrightness(windowWidth, windowHeight, windowX, windowY, otherImage))

            # put them into the lists
            imageSignatures.append(imageSignature)
            otherImageSignatures.append(otherImageSignature)

            windowIndex += 1



    # now we must normalize such that sum of brightness is the same
    normalizeBrightness(imageSignatures, otherImageSignatures)

    # now that they are normalized, we must formulate the linprog input

    # first identify the function to minimize
    distances = []
    for imageSignature in imageSignatures:
        for otherImageSignature in otherImageSignatures:
            distances.append(windowDistance(windowWidth, windowHeight, imageSignature[0], otherImageSignature[0]))
    
    # now the constraints and bounds
    constraints = []
    bounds = []

    # now for Fij >= 0
    windowCount = len(imageSignatures)

    for index in range(windowCount * windowCount):
        positiveFlow = ([0] * index) + [-1] + ([0] * (windowCount * windowCount - 1 - index))
        constraints.append(positiveFlow)
        bounds.append(0)

    # now sum over j of Fij = wi for every i 

    for i in range(windowCount):
        lte1 = ([0] * (i * windowCount)) + ([1] * windowCount) + ([0] * (windowCount - i - 1) * (windowCount))
        gte1 = ([0] * (i * windowCount)) + ([-1] * windowCount) + ([0] * (windowCount - i - 1) * (windowCount))

        lte2 = []
        gte2 = []

        for number in range(windowCount * windowCount):
            if number % windowCount == i:
                lte2.append(1)
                gte2.append(-1)
            else:
                lte2.append(0)
                gte2.append(0)

        imageW = imageSignatures[i][1]
        otherImageW = otherImageSignatures[i][1]

        constraints.append(lte1)
        bounds.append(imageW)

        constraints.append(gte1)
        bounds.append(-imageW)

        constraints.append(lte2)
        bounds.append(otherImageW)

        constraints.append(gte2)
        bounds.append(-otherImageW)

    res = linprog(distances, A_ub=constraints, b_ub=bounds)
    return res.fun

def normalizeBrightness(imageSignatures, otherImageSignatures):
    goalBrightness = 1

    # calculate total brightnesses
    imageBrightness = sum([brightness for (_, brightness) in imageSignatures])
    otherImageBrightness = sum([brightness for (_, brightness) in otherImageSignatures])

    scalingFactor = goalBrightness / otherImageBrightness
    newOtherImageSignatures = [(index, brightness * scalingFactor) for (index, brightness) in otherImageSignatures]
    otherImageSignatures.clear()
    otherImageSignatures.extend(newOtherImageSignatures)

    scalingFactor = goalBrightness / imageBrightness
    newImageSignatures = [(index, brightness * scalingFactor) for (index, brightness) in imageSignatures]
    imageSignatures.clear()
    imageSignatures.extend(newImageSignatures)



def calculateBrightness(width, height, i, j, image):
    brightness = 0

    for y in range(height * j, height * (j + 1)):
        for x in range(width * i, width * (i + 1)):
            brightness += image[y * width + x]

    return brightness + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d')
    parser.add_argument('-q')
    parser.add_argument('-l1')
    parser.add_argument('-l2')
    parser.add_argument('-o')
    parser.add_argument('-width', default="7")
    parser.add_argument('-height', default="7")


    args = parser.parse_args()

    inputFilePath = args.d
    queryFilePath = args.q
    inputLabelsPath = args.l1
    queryLabelsPath = args.l2
    outputPath = args.o
    inputWidth = int(args.width)
    inputHeight = int(args.height)

    # read images
    inputImages = readImages(inputFilePath)
    queryImages = readImages(queryFilePath)

    # read labels
    inputLabels = readLabels(inputLabelsPath)
    queryLabels = readLabels(queryLabelsPath)

    queryIndex = 0

    totalEmdCorrectness = 0
    totalManhattanCorrectness = 0

    # for every query image
    for queryImage in queryImages:
        queryClass = queryLabels[queryIndex]
        emdCorrectGuesses = 0
        manhattanCorrectGuesses = 0


        # find 10 nearest neighbors with manhattan distance metric
        emdNearest = kNearestNeighbors(queryImage, inputImages, 10, partial(earthMoverDistance, inputWidth, inputHeight))
        manhattanNearest = kNearestNeighbors(queryImage, inputImages, 10, manhattanDistance)

        # for every neighbor, see if it is in the correct class
        for neighborId in emdNearest:
            if queryClass == inputLabels[neighborId]:
                emdCorrectGuesses += 1

        for neighborId in manhattanNearest:
            if queryClass == inputLabels[neighborId]:
                manhattanCorrectGuesses += 1

        totalEmdCorrectness += emdCorrectGuesses / 10
        totalManhattanCorrectness += manhattanCorrectGuesses / 10


        queryIndex += 1


    outputFile = open(outputPath, 'w') 
    outputFile.write("Average Correct Search Results EMD: " + str(totalEmdCorrectness) + "\n")
    outputFile.write("Average Correct Search Results Manhattan: " + str(totalManhattanCorrectness) + "\n")






if __name__ == "__main__":
    main()
