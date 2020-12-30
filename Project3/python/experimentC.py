from os import sched_get_priority_min
from subprocess import call
import re
import matplotlib.pyplot as plt
import numpy as np

from timeit import default_timer as timer

def main():
    X = [28, 14, 7]
    Y = [28, 14, 7]
    Z = []
    T = []

    # try out all combinations of width/height in [28, 14, 7]
    for width in X:
        currZ = []
        currT = []
        for height in Y:
            start = timer()
            # run emd.py and collect results
            call(["python3", "./emd.py", 
                  "-d", "../originalSpace/verySmallData",
                  "-q", "../originalSpace/tinyData",
                  "-l1", "../originalSpace/verySmallLabels",
                  "-l2", "../originalSpace/tinyLabels",
                  "-o", "./temp",
                  "-width", str(width),
                  "-height", str(height)])
            end = timer()

            # collect output from file and store into directory
            outputFile = open("./temp", "r")
            lines = outputFile.readlines()
            # get results
            emdString = lines[0][re.search(r"\d", lines[0]).start():]
            manhattanString = lines[1][re.search(r"\d", lines[1]).start():]

            # since manhattan is consistent, keep the ratio instead
            emdCorrectness = float(emdString)
            manhattanCorrectness = float(manhattanString)
            emdToManhattanRatio = emdCorrectness / manhattanCorrectness


            currZ.append(emdToManhattanRatio)
            currT.append(end - start)

        Z.append(currZ)
        T.append(currT)


    # make x and y appropriate 
    meshX, meshY = np.meshgrid(X, Y)

    # now that we have all the data, plot ratio of correctness
    ax = plt.axes(projection='3d')
    ax.plot_surface(meshX, meshY, np.array(Z), rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_title('EMD / Manhattan correctness')
    ax.set_xlabel('width')
    ax.set_ylabel('height')
    plt.show()


    # also plot time
    ax = plt.axes(projection='3d')
    ax.plot_surface(meshX, meshY, np.array(T), rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_title('Runtime')
    ax.set_xlabel('width')
    ax.set_ylabel('height')
    plt.show()
                


if __name__ == "__main__":
    main()
