#include "../headers/Aux.h"
#include <iostream>

using namespace std;

int main(int argc, char **argv){
    string *inputFileOGPath, *inputFileNewPath, *queryFileOGPath, *queryFileNewPath, *kStr, *LStr, *outputFilePath;
    unsigned int numberOfImages, pixels, k, L, numberOfQueries;
    vector<Image *> *imagesOG, *imagesNew, *queryImagesOG, *queryImagesNew;
    fstream *inputFileOG, *inputFileNew, *queryFileOG, *queryFileNew;

    //Get all arguments
    inputFileOGPath = getOptionValue("-d", argc, argv);
    if(inputFileOGPath == NULL){
        cout << "Please enter the input file original space path:" << endl;

        inputFileOGPath = new string();
        getline(cin, *inputFileOGPath);
    }

    inputFileNewPath = getOptionValue("-i", argc, argv);
    if(inputFileNewPath == NULL){
        cout << "Please enter the input file new space path:" << endl;

        inputFileNewPath = new string();
        getline(cin, *inputFileNewPath);
    }

    queryFileOGPath = getOptionValue("-q", argc, argv);
    if(queryFileOGPath == NULL){
        cout << "Please enter the query file original space path:" << endl;

        queryFileOGPath = new string();
        getline(cin, *queryFileOGPath);
    }

    queryFileNewPath = getOptionValue("-s", argc, argv);
    if(queryFileNewPath == NULL){
        cout << "Please enter the query file new space path:" << endl;

        queryFileNewPath = new string();
        getline(cin, *queryFileNewPath);
    }

    kStr = getOptionValue("-k", argc, argv);
    if(kStr == NULL){
        k = 4;
    } else{
        k = stoi(*kStr);
        delete kStr;
    }

    LStr = getOptionValue("-L", argc, argv);
    if(LStr == NULL){
        L = 5;
    } else{
        L = stoi(*LStr);
        delete LStr;
    }

    outputFilePath = getOptionValue("-o", argc, argv);
    if(outputFilePath == NULL){
        cout << "Please enter the ouput file path:" << endl;

        outputFilePath = new string();
        getline(cin, *outputFilePath);
    }

    // Open all files 
    imagesOG = openTrainFile(inputFileOGPath, pixels, numberOfImages, &inputFileOG);
    delete inputFileOGPath; 

    parseImage(NULL, 0);

    // Initialize LSH
    LSH* myLSH = initializeLSH(k, pixels, L, imagesOG);

    imagesNew = openTrainFile(inputFileNewPath, pixels, numberOfImages, &inputFileNew);
    delete inputFileNewPath;

    parseImage(NULL, 0);

    queryImagesOG = openTrainFile(queryFileOGPath, pixels, numberOfQueries, &queryFileOG);
    delete queryFileOGPath;

    parseImage(NULL, 0);

    queryImagesNew = openTrainFile(queryFileNewPath, pixels, numberOfQueries, &queryFileNew);
    delete queryFileNewPath;

    parseImage(NULL, 0);

    //Output function
    calculateOutput(outputFilePath, numberOfImages, imagesOG, imagesNew, queryImagesOG, queryImagesNew, myLSH, k, 1);

    //clean query images    
    // for(Image *image : *queryImages){
    //     delete image;
    // }
    // delete queryImages;
}