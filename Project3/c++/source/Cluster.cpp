#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <arpa/inet.h>
#include "../headers/Aux.h"
#include "../headers/Hashtable.h"
#include "../headers/Image.h"
#include "../headers/LSH.h"
#include "../headers/HyperCube.h"
#include <cmath>


using namespace std;

int main(int argc, char **argv){
    string *inputFilePath, *configFilePath, *outputFilePath, *methodStr, *osFilePath, *nsFilePath, *nnFilePath;
    bool complete = false;
    unsigned int magic_number, numberOfImages, pixels, k, N, numberOfQueries;
    int numberOfClusters, kLSH, MHyperCube, kHyperCube, probesHyperCube, L;
    double R;
    vector<Image *> *images, *queryImages, *osImages, *nsImages;
    vector<set<Image *> *> *NNClusters;

    LSH *myLSH = NULL;
    HyperCube *myHyperCube = NULL;

    osFilePath = getOptionValue("-d", argc, argv);
    if(osFilePath == NULL){
        cout << "Please enter the original space input file path:" << endl;

        osFilePath = new string();
        getline(cin, *osFilePath);
    }

    nsFilePath = getOptionValue("-i", argc, argv);
    if(nsFilePath == NULL){
        cout << "Please enter the new space input file path:" << endl;

        nsFilePath = new string();
        getline(cin, *nsFilePath);
    }

    nnFilePath = getOptionValue("-n", argc, argv);
    if(nnFilePath == NULL){
        cout << "Please enter the neural network clusters file path:" << endl;

        nnFilePath = new string();
        getline(cin, *nnFilePath);
    }

    configFilePath = getOptionValue("-c", argc, argv);
    if(configFilePath == NULL){
        cout << "Please enter the config file path:" << endl;

        configFilePath = new string();
        getline(cin, *configFilePath);
    }

    outputFilePath = getOptionValue("-o", argc, argv);
    if(outputFilePath == NULL){
        cout << "Please enter the output file path:" << endl;

        outputFilePath = new string();
        getline(cin, *outputFilePath);
    }


    //get info from config file
    readConfigFile(configFilePath, &numberOfClusters, &L, &kLSH, &MHyperCube, &kHyperCube, &probesHyperCube);

    delete configFilePath;

    //read the input file
    fstream *osFile;
    osImages = openTrainFile(osFilePath, pixels, numberOfImages, &osFile);

    //reset idCounter in parseImage
    parseImage(NULL, 0);

    delete osFilePath;

    osFile->close();

    delete osFile;

    //read the input file
    fstream *nsFile;
    nsImages = openTrainFile(nsFilePath, pixels, numberOfImages, &nsFile);

    delete nsFilePath;

    nsFile->close();

    delete nsFile;

    //now open the output file in order to write to it
    ofstream *outputFile = new ofstream(outputFilePath->c_str(), ios::out);
    delete outputFilePath;


     //actually assign memory for the method
    methodStr = new string();

    //set the method
    methodStr->assign("Classic");

    // write output for new space
    *outputFile << "NEW SPACE" << endl;

    runClusteringAlgorithm(nsImages, osImages, true, numberOfClusters, methodStr, outputFile, myLSH, myHyperCube, MHyperCube, probesHyperCube, complete);

    // write output for original space
    *outputFile << "ORIGINAL SPACE" << endl;

    runClusteringAlgorithm(osImages, nsImages, false, numberOfClusters, methodStr, outputFile, myLSH, myHyperCube, MHyperCube, probesHyperCube, complete);


    *outputFile << "CLASSES AS CLUSTERS" << endl;


    // calculate and write silhouette for classes as clusters
    NNClusters = readClustersFromFile(nnFilePath, osImages);
    vector<Image *> *NNCentroids = centroidize(NNClusters);

    vector<double> *silhouettes = calculateSilhouettes(NNCentroids, NNClusters);

    *outputFile << "Silhouette: [";

    const auto* sep = "";
    for(double currSil : *silhouettes) {
        *outputFile << sep << currSil;
        sep = ", ";
    }

    *outputFile << "]" << endl;

    delete silhouettes;


    *outputFile << "Value of Objective Function: " << objectiveFunction(osImages, NNCentroids) << endl;


    // //see what methdos we need to run
    // if(methodStr == NULL){ // all algorithms

    //     //actually assign memory for the method
    //     methodStr = new string();

    //     //set the method
    //     methodStr->assign("Classic");

    //     //first output for classic
    //     *outputFile << "Algorithm: Lloyds" << endl;

    //     runClusteringAlgorithm(images, numberOfClusters, methodStr, outputFile, myLSH, myHyperCube, MHyperCube, probesHyperCube, complete);

    //     //build LSH database
    //     myLSH = initializeLSH(kLSH, pixels, L, images);


    //     //set the method
    //     methodStr->assign("LSH");

    //     //output
    //     *outputFile << "Algorithm: Range Search LSH" << endl;

    //     runClusteringAlgorithm(images, numberOfClusters, methodStr, outputFile, myLSH, myHyperCube, MHyperCube, probesHyperCube, complete);

    //     delete myLSH;

    //     free(Hashtable::mPowerModulos);

    //     //build HyperCube database
    //     myHyperCube = initializeHyperCube(kHyperCube, pixels, images);


    //     //set the method
    //     methodStr->assign("Hypercube");

    //     //output
    //     *outputFile << "Algorithm: Range Search Hypercube" << endl;

    //     runClusteringAlgorithm(images, numberOfClusters, methodStr, outputFile, myLSH, myHyperCube, MHyperCube, probesHyperCube, complete);

    //     delete myHyperCube;

    // } else if(*methodStr == "LSH"){ //lsh only
    //     //build LSH database
    //     myLSH = initializeLSH(kLSH, pixels, L, images);

    //     //output
    //     *outputFile << "Algorithm: Range Search LSH" << endl;

    //     runClusteringAlgorithm(images, numberOfClusters, methodStr, outputFile, myLSH, myHyperCube, MHyperCube, probesHyperCube, complete);

    //     delete myLSH;
    // } else if(*methodStr == "Hypercube"){ //hypercube only
    //     //build HyperCube database
    //     myHyperCube = initializeHyperCube(kHyperCube, pixels, images);

    //     //output
    //     *outputFile << "Algorithm: Range Search Hypercube" << endl;

    //     runClusteringAlgorithm(images, numberOfClusters, methodStr, outputFile, myLSH, myHyperCube, MHyperCube, probesHyperCube, complete);

    //     delete myHyperCube;

    // } else { //classic only
    //     *outputFile << "Algorithm: Lloyds" << endl;

    //     runClusteringAlgorithm(images, numberOfClusters, methodStr, outputFile, myLSH, myHyperCube, MHyperCube, probesHyperCube, complete);
    // }


    if(methodStr != NULL){
        delete methodStr;
    }

    outputFile->close();
    delete outputFile;


    for(Image *image : *nsImages){
        delete image;
    }
    delete nsImages;

    for(Image *image : *osImages){
        delete image;
    }
    delete osImages;

    free(Hashtable::mPowerModulos);

    return 0;

}