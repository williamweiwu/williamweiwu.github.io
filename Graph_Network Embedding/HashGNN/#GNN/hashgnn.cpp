#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <math.h>
#include <algorithm>
#include <time.h> 
#include <fstream>
#include <deque>
#include <vector>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#define MAX_STRING 100
#define MAX_NODE_NUM 1000000000
#define MAX_FEATURE_NUM 10000000	// warning: large MAX_FEATURE_NUM causes segmenation default
#define MAX_PARAMETER_NUM 300
#define PRIME_NUM 348513

using namespace std;


char networkFile[MAX_STRING], featureFile[MAX_STRING], embeddingFile[MAX_STRING], timeFile[MAX_STRING];


// return unsigned int* primeArray
void readRandomPrimes(string primesFile, unsigned int* primeArray) {
	string line;
	int i=0;
	ifstream  fin(primesFile);
	if(!fin.is_open()) {
		cout<<"primes file not found!"<<endl;
	}
	while(getline(fin, line) && i<PRIME_NUM) {
		primeArray[i] = stoi(line);
		i++;
	}
	fin.close();
}

void generateRandomPrimes(unsigned int* primeArray, int k, int iteration, 
	unsigned int ***hashU1ParametersThreeArrays, unsigned int ***hashU2ParametersThreeArrays, 
	unsigned int ***hashVParametersThreeArrays, unsigned int *divisor) {
	
	srand(1);
	
	int hashNum = k;
	for(int iR = 1; iR < iteration+1; iR++) { 
		// coefficients
		unsigned int **hashU1ParametersTwoArrays = new unsigned int*[hashNum];
		unsigned int **hashU2ParametersTwoArrays = new unsigned int*[hashNum];
		unsigned int **hashVParametersTwoArrays = new unsigned int*[hashNum];

		for(int iK=0; iK < hashNum; iK++) {

			unsigned int *hashU1ParametersOneArray = new unsigned int[2];
			unsigned int *hashU2ParametersOneArray = new unsigned int[2];
			unsigned int *hashVParametersOneArray = new unsigned int[2];

			for(int iHash=0; iHash< 2; iHash++) {

				hashU1ParametersOneArray[iHash] = primeArray[rand()%PRIME_NUM];
				hashU2ParametersOneArray[iHash] = primeArray[rand()%PRIME_NUM];
				hashVParametersOneArray[iHash] = primeArray[rand()%PRIME_NUM];

			}

			hashU1ParametersTwoArrays[iK] = hashU1ParametersOneArray;
			hashU2ParametersTwoArrays[iK] = hashU2ParametersOneArray;
			hashVParametersTwoArrays[iK] = hashVParametersOneArray;

		}

		hashU1ParametersThreeArrays[iR] = hashU1ParametersTwoArrays;
		hashU2ParametersThreeArrays[iR] = hashU2ParametersTwoArrays;
		hashVParametersThreeArrays[iR] = hashVParametersTwoArrays;
		
	}
	divisor[0] = primeArray[PRIME_NUM-1];
}

// read network
// return int **network, int *adjNum
// index starts at 0
void readNetwork(char* adjListFile, int networkSize, int **network, int *adjNum) {
	string line;
	ifstream  fin(adjListFile);
	if(!fin.is_open()) {
		cout<<"adjacent list file not found!"<<endl;
	}
	vector<string> adjlist;
	int *nodes = NULL;
	for(int iRow=0; iRow< networkSize; iRow++) {
		getline(fin, line);
		if(line.empty()) {
			adjNum[iRow] = 0;
		} else {
			boost::split(adjlist, line, boost::is_any_of(" "), boost::token_compress_on);
	
			nodes = new int[adjlist.size()];
			for(int iNode=0; iNode < adjlist.size(); iNode++) {
				nodes[iNode] = stoi(adjlist.at(iNode));
			}
			network[iRow] = nodes;
			adjNum[iRow] = adjlist.size();
		}

		adjlist.clear();
	}	
	fin.close();
}

// read feature
// features are seperated by one whitespace
// return unsigned int **tmpFeatureTwoArrays, int *tmpFeatureNumOneArray, int networkSize
int readFeatures(char* featureFile, unsigned int **tmpFeatureTwoArrays, int *tmpFeatureNumOneArray) {
	string line;
	ifstream  fin(featureFile);
	if(!fin.is_open()) {
		cout<<"feature file not found!"<<endl;
	}
	vector<string> featureVector;
	unsigned int *featureArray = NULL;
	int nodeNum = 0;
	while(getline(fin, line)) {

		if(line.empty()) {
			//featureArray = NULL;
			tmpFeatureNumOneArray[nodeNum] = 1;
			featureArray = new unsigned int[1];
			featureArray[0] = MAX_FEATURE_NUM;
		} else {
			boost::split(featureVector, line, boost::is_any_of(" "), boost::token_compress_on);
			featureArray = new unsigned int[featureVector.size()];
			for(int iFeature=0; iFeature < featureVector.size(); iFeature++) {
				featureArray[iFeature]= stoi(featureVector.at(iFeature));
			}
			tmpFeatureNumOneArray[nodeNum] = featureVector.size();
		}

		tmpFeatureTwoArrays[nodeNum] = featureArray;
		//tmpFeatureNumOneArray[nodeNum] = featureVector.size();
		nodeNum++;
		featureVector.clear();

	}
	fin.close();

	return nodeNum;
}


void minHash(unsigned int **hashParameters, unsigned int *divisor, int k, unsigned int *features, int featureNum, unsigned int* minHashValues, unsigned int* minHashResults){
	
	// row: feature
	// column: parameter
	// the operation of minhash is frequent, so static arrays are preferred for effeciency.

	if(features[0] == MAX_FEATURE_NUM) {
		for(int iColumn = 0; iColumn < k; iColumn++) {	
			minHashValues[iColumn] = MAX_FEATURE_NUM;
			minHashResults[iColumn] = divisor[0] + 1;
		}
		return;
	}

	//unsigned int result[MAX_FEATURE_NUM][MAX_PARAMETER_NUM];
	unsigned int **result =  new unsigned int *[featureNum];
	for(int iRow = 0; iRow< featureNum; iRow++) {	
		for(int jColumn = 0; jColumn < k; jColumn++) {
			result[iRow][jColumn] = (features[iRow]* hashParameters[jColumn][0] + hashParameters[jColumn][1]) % divisor[0];
		}
	}
		
	unsigned int minValue;
	int minIndex;
	for(int iColumn = 0; iColumn < k; iColumn++) {	
		minValue = result[0][iColumn];
		minIndex = 0;
		for(int jRow = 1; jRow < featureNum; jRow++) {
			if(minValue > result[jRow][iColumn]) {
				minValue = result[jRow][iColumn];
				minIndex = jRow;
			}
		}
		minHashValues[iColumn] = features[minIndex];
		minHashResults[iColumn] = minValue;
	}
	for (int iRow=0; iRow<featureNum; iRow++) {
        delete[] result[iRow];
    }
    delete[] result;
}

void minHash(unsigned int hashParameter1, unsigned int hashParameter2, unsigned int divisor, unsigned int *features, int featureNum, unsigned int* minHashValue, unsigned int* minHashResult){
	
	// row: feature
	// column: parameter
	// the operation of minhash is frequent, so static arrays are preferred for effeciency.

	if(features[0] == MAX_FEATURE_NUM) {

		*minHashValue = MAX_FEATURE_NUM;
		*minHashResult = divisor + 1;
		return;
	}

	unsigned int *result =  new unsigned int [featureNum];
	for(int iF=0; iF<featureNum; iF++) {

		result[iF] = (features[iF]* hashParameter1 + hashParameter2) % divisor;

	}

	unsigned int minValue = result[0];
	int minIndex = 0;

	for(int iF=1; iF<featureNum; iF++) {

		if(minValue > result[iF]) {

			minValue = result[iF];
			minIndex = iF;
		}
	}
	minHashValue[0] = features[minIndex];
	minHashResult[0] = minValue;
	delete[] result;

}

void mergeMessages(unsigned int *features, int featureNum, unsigned int *firstComponent, int firstComponentSize, unsigned int *mergedFeatures) {
	
	for(int iFeature = 0; iFeature < featureNum; iFeature++) {
		*mergedFeatures++ = *features++;
	}
	for(int iFeature = 0; iFeature < firstComponentSize; iFeature++) {
		*mergedFeatures++ = *firstComponent++;
	}
}

void passMessages(int **network, int *adjNum, int networkSize, int iteration, int k, 
	unsigned int **featureTwoArrays, int *featureNumOneArray, 
	unsigned int ***hashU1ParametersThreeArrays, unsigned int ***hashU2ParametersThreeArrays, unsigned int ***hashVParametersThreeArrays, 
	unsigned int *divisor, unsigned int **fingerprints) {

	unsigned int *features = NULL;
	unsigned int *tmpMergedMessages = NULL;
	unsigned int *sumMessages = NULL;

	unsigned int *minHashValue = NULL;
	unsigned int *minHashResult = NULL;

	int featureNum=0;
	unsigned int *u1Message = NULL;
	unsigned int *u1MinHashResult = NULL;
	unsigned int *u2Message = NULL;
	unsigned int *u2MinHashResult = NULL;
	unsigned int *vMessage = NULL;
	unsigned int *vMinHashResult = NULL;
	unsigned int **vMessages = new unsigned int *[networkSize];


	for(int iNode=0; iNode<networkSize; iNode++) {

		fingerprints[iNode] = new unsigned int[k];
	}

	for(int iR=1; iR<iteration+1; iR++) {

		for(int iK=0; iK<k; iK++) {

			for(int iNode=0; iNode<networkSize; iNode++) {

				if(iR==1) {
					features = featureTwoArrays[iNode];
					featureNum = featureNumOneArray[iNode];
				} else {
					features = fingerprints[iNode];
					featureNum = k;
				}	

				vMessage = new unsigned int[1];
				vMinHashResult = new unsigned int[1];
				minHash(hashVParametersThreeArrays[iR][iK][0], hashVParametersThreeArrays[iR][iK][1], divisor[0], features, featureNum, vMessage, vMinHashResult);
				
				vMessages[iNode] = vMessage;

				delete[] vMinHashResult;
			}


			for(int iNode=0; iNode<networkSize; iNode++) {

				if(iR==1) {
					features = featureTwoArrays[iNode];
					featureNum = featureNumOneArray[iNode];
				} else {
					features = fingerprints[iNode];
					featureNum = k;
				}	

				u1Message = new unsigned int[1];
				u1MinHashResult = new unsigned int[1];
				minHash(hashU1ParametersThreeArrays[iR][iK][0], hashU1ParametersThreeArrays[iR][iK][1], divisor[0], features, featureNum, u1Message, u1MinHashResult);


				sumMessages = new unsigned int [adjNum[iNode]];
				for(int iAdj=0; iAdj<adjNum[iNode]; iAdj++) {

					sumMessages[iAdj] = vMessages[network[iNode][iAdj]][0];
				}
				u2Message = new unsigned int[1];
				u2MinHashResult = new unsigned int[1];
				minHash(hashU2ParametersThreeArrays[iR][iK][0], hashU2ParametersThreeArrays[iR][iK][1], divisor[0], sumMessages, adjNum[iNode], u2Message, u2MinHashResult);

				if(u1MinHashResult[0] <= u2MinHashResult[0]) {

					fingerprints[iNode][iK] = u1Message[0];
				} else {

					fingerprints[iNode][iK] = u2Message[0];
				}

				delete[] u1Message;
				delete[] u1MinHashResult;
				delete[] u2Message;
				delete[] u2MinHashResult;
				delete[] sumMessages;
			}
			
			for(int iNode=0; iNode<networkSize; iNode++) {

				delete[] vMessages[iNode];
			}

		}

	}
	delete[] vMessages;

}

void output(unsigned int **fingerprints, int networkSize, int k, double elapsedTime) {
	ofstream fout(embeddingFile);
	if(!fout.is_open()) {
		cout<<"fail to open embedding file!"<<endl;
	}
	for(int iNetwork = 0; iNetwork< networkSize; iNetwork++) {
		for(int iK=0; iK< k; iK++) {
			fout<< fingerprints[iNetwork][iK]<< " ";
		} 
		fout<<endl;
	}
	fout.close();
	ofstream fout1(timeFile);
	if(!fout1.is_open()) {
		cout<<"fail to open time file!"<<endl;
	}
	fout1<< elapsedTime<< endl;
	fout1.close();
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			cout<<"Argument missing for "<< str<<endl;
			exit(1);
		}
		return a;
	}
	return -1;
}

int main(int argc, char **argv) {
	
	int k = 0;
	int iteration = 0;
	
	int i;
	if (argc == 1) {
		cout<<"Deep Recursive Hashing"<<endl;
		cout<<"Options:"<<endl;
		cout<<"Parameters for algorithm:"<<endl;
		cout<<"-network <file>"<<endl;
		cout<<"Network data of adjlist from <file> for structure"<<endl;
		cout<<"-feature <file>"<<endl;
		cout<<"Feature data from <file> for contents"<<endl;
		cout<<"-hashdim <int>"<<endl;
		cout<<"Number of dimension of network embeddings"<<endl;
		cout<<"-iteration <int>"<<endl;
		cout<<"iteration of neural network"<<endl;
		cout<<"-embedding <file>"<<endl;
		cout<<"<file> to save the embeddings"<<endl;
		cout<<"-time <file>"<<endl;
		cout<<"<file> to save running time"<<endl;
		return 0;
	}
	if ((i = ArgPos((char *)"-network", argc, argv)) > 0) {
		strcpy(networkFile, argv[i + 1]);
	}
	if ((i = ArgPos((char *)"-feature", argc, argv)) > 0) {
		strcpy(featureFile, argv[i + 1]);
	}
	if ((i = ArgPos((char *)"-hashdim", argc, argv)) > 0) {
		k = atoi(argv[i + 1]);
	}
	if ((i = ArgPos((char *)"-iteration", argc, argv)) > 0) {
		iteration = atoi(argv[i + 1]);
	}
	if ((i = ArgPos((char *)"-embedding", argc, argv)) > 0) {
		strcpy(embeddingFile, argv[i + 1]);
	}
	if ((i = ArgPos((char *)"-time", argc, argv)) > 0) {
		strcpy(timeFile, argv[i + 1]);
	}
	
	cout<< "hashdim: "<<k<< ", iteration: "<< iteration <<endl;
	
	// parameters for hash functions
	unsigned int *primeArray = new unsigned int[PRIME_NUM];
	unsigned int *divisor = new unsigned int[1];
	readRandomPrimes("primes.txt", primeArray);

	unsigned int ***hashU1ParametersThreeArrays = new unsigned int**[iteration+1];	
	unsigned int ***hashU2ParametersThreeArrays = new unsigned int**[iteration+1];
	unsigned int ***hashVParametersThreeArrays = new unsigned int**[iteration+1];

	generateRandomPrimes(primeArray, k, iteration, 
		hashU1ParametersThreeArrays, hashU2ParametersThreeArrays, hashVParametersThreeArrays, divisor);

	delete[] primeArray;
			
	// features on each node	
	unsigned int **tmpFeatureTwoArrays = new unsigned int*[MAX_NODE_NUM];
	int *tmpFeatureNumOneArray = new int[MAX_NODE_NUM];
	int networkSize = readFeatures(featureFile, tmpFeatureTwoArrays, tmpFeatureNumOneArray);	
	unsigned int **featureTwoArrays = new unsigned int*[networkSize];
	int *featureNumOneArray = new int[networkSize];
	for(int iNode = 0; iNode < networkSize; iNode++) {
		featureTwoArrays[iNode] = new unsigned int[tmpFeatureNumOneArray[iNode]];
		for(int iFeature = 0; iFeature < tmpFeatureNumOneArray[iNode]; iFeature++) {
			featureTwoArrays[iNode][iFeature] = tmpFeatureTwoArrays[iNode][iFeature];
		}
		featureNumOneArray[iNode] = tmpFeatureNumOneArray[iNode];
	}
	//delete temporary memory
	for(int iNode = 0; iNode < MAX_NODE_NUM; iNode++) {
		delete[] tmpFeatureTwoArrays[iNode];
	}
	delete[] tmpFeatureTwoArrays;
	delete[] tmpFeatureNumOneArray;
	
	// network structure
	int **network = new int*[networkSize];
	for(int iNode=0; iNode< networkSize; iNode++) {
		network[iNode] = NULL;
	}
	int *adjNum = new int[networkSize];
	readNetwork(networkFile, networkSize, network, adjNum);

	clock_t timeBegin, timeEnd;
	double elapsedTime;
	unsigned int **fingerprints = new unsigned int *[networkSize];
	
	timeBegin = clock();	

	passMessages(network, adjNum, networkSize, iteration, k, featureTwoArrays, featureNumOneArray, hashU1ParametersThreeArrays, hashU2ParametersThreeArrays, hashVParametersThreeArrays, divisor, fingerprints);


	timeEnd = clock();
	elapsedTime = (double(timeEnd - timeBegin))/CLOCKS_PER_SEC;
	cout<< "elapsed time: "<< elapsedTime << endl;
	
	output(fingerprints, networkSize,  k, elapsedTime);
	
	return 0;
	
}
