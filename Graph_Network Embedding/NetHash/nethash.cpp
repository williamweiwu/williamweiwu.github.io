#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <math.h>
#include <algorithm>
#include <time.h> 
#include <fstream>
#include <deque>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#define PPT_SIZE 1000000000
#define MAX_STRING 100
#define MAX_NODE_NUM 1000000000
#define MAX_FEATURE_NUM 1000000	// warning: large MAX_FEATURE_NUM causes segmenation default
#define MAX_PARAMETER_NUM 300
#define PRIME_NUM 348513

using namespace std;


char networkFile[MAX_STRING], featureFile[MAX_STRING], embeddingFile[MAX_STRING], timeFile[MAX_STRING];
int ppt[PPT_SIZE][3];

struct PPTparameters {
	int lengthOfPPT;
	int numberOfLeaves;
};


struct ElementInQueue {
	unsigned int *minHashValues;
	int iNode;
};


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

// return unsigned int ***hashParametersThreeArrays, unsigned int **divisorsTwoArrays, int *parameterNum
void generateRandomPrimes(unsigned int* primeArray, int k, float degreeEntropy, int depth, 
	unsigned int ***hashParametersThreeArrays, unsigned int **divisorsTwoArrays, int *parameterNum) {
	
	srand(time(NULL));
	
	// in each setting, (prob, depth), there are int(rint(k* exp(-iDepth * degreeEntropy))) hash functions
	int hashNum = 0;
	for(int iDepth = 0; iDepth < depth+1; iDepth++) { 
		hashNum = max(1, int(rint(k * exp(-iDepth * degreeEntropy))));
		// coefficients
		unsigned int **hashParametersTwoArrays = new unsigned int*[hashNum];
		for(int iK=0; iK < hashNum; iK++) {
			unsigned int *hashParametersOneArray = new unsigned int[2];
			for(int iHash=0; iHash< 2; iHash++) {
				hashParametersOneArray[iHash] = primeArray[rand()%PRIME_NUM];
			}
			hashParametersTwoArrays[iK] = hashParametersOneArray;
		}
		hashParametersThreeArrays[iDepth] = hashParametersTwoArrays;
		
		// divisors
		unsigned int *divisorsOneArray = new unsigned int[hashNum];
		for(int iDivisors = 0; iDivisors< hashNum; iDivisors++) {
			divisorsOneArray[iDivisors] = primeArray[PRIME_NUM-1-iDivisors];
		}
		divisorsTwoArrays[iDepth] = divisorsOneArray;
		
		// number of parameters
		parameterNum[iDepth] = hashNum;
	}
}

// read network
// return int **network, int *adjNum
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
		nodeNum++;
		featureVector.clear();
	}
	fin.close();

	return nodeNum;
}


struct PPTparameters buildParentPointerTree(int **network, int *adjNum, int networkSize, int nodeId, int depth, int ppt[][3]) {
		
	struct PPTparameters pptParameters;
	
	// node as root of a ppt
	int pptSize = 1;
	ppt[0][0] = nodeId;
	ppt[0][1] = -1;
	ppt[0][2] = 0;

	if(!network[nodeId]) {
		pptParameters.lengthOfPPT = 1;
		pptParameters.numberOfLeaves = 1;
		return pptParameters;
	}
	
	
	// in each level, point the leftmost child and rightmost child, respectively
	int leftmost = 0;
	int rightmost = 0;
	// current parent
	int iParent = -1;
	// build a ppt for each node from Depth = 1, and root is at Depth = 0
	int currentNode;
	int *children = NULL;
	
	for(int d = 1; d<= depth; d++) {
		for(int iNode = leftmost; iNode <= rightmost; iNode++) {
			iParent++;
			currentNode = ppt[iNode][0];
			children = network[currentNode];			
			for(int iChildren=0; iChildren < adjNum[currentNode]; iChildren++) {
				ppt[pptSize][0] = children[iChildren];
				ppt[pptSize][1] = iParent;
				ppt[pptSize][2] = d;
				pptSize++;
			}
		}
		leftmost = rightmost+1;
		rightmost = pptSize -1;
		
	}
	
	// numberOfLeaves
	pptParameters.lengthOfPPT = pptSize;
	pptParameters.numberOfLeaves = rightmost - leftmost +1;
	return pptParameters;
}


void minHash(unsigned int **hashParameters, unsigned int *divisors, int parameterNum, unsigned int *features, int featureNum, unsigned int* minHashValues){
	
	// row: feature
	// column: parameter
	// the operation of minhash is frequent, so static arrays are preferred for effeciency.
	
	if(features[0] == MAX_FEATURE_NUM) {
		for(int iColumn = 0; iColumn < parameterNum; iColumn++) {	
			minHashValues[iColumn] = MAX_FEATURE_NUM;
		}
		return;
	}
	
	unsigned int result[MAX_FEATURE_NUM][MAX_PARAMETER_NUM];
	for(int iRow = 0; iRow< featureNum; iRow++) {	
		for(int jColumn = 0; jColumn < parameterNum; jColumn++) {
			
			if(features[iRow] == MAX_FEATURE_NUM) {
				result[iRow][jColumn] = divisors[0] + 1;
			} else {
				result[iRow][jColumn] = (features[iRow]* hashParameters[jColumn][0] + hashParameters[jColumn][1]) % divisors[jColumn];
			}
			
		}
	}
		
	unsigned int minValue;
	int minIndex;
	for(int iColumn = 0; iColumn < parameterNum; iColumn++) {	
		minValue = result[0][iColumn];
		minIndex = 0;
		for(int jRow = 1; jRow < featureNum; jRow++) {
			if(minValue > result[jRow][iColumn]) {
				minValue = result[jRow][iColumn];
				minIndex = jRow;
			}
		}
		minHashValues[iColumn] = features[minIndex];
	}
}

void mergeArrays(unsigned int *features, int featureNum, unsigned int *firstComponent, int firstComponentSize, unsigned int *mergedFeatures) {
	
	for(int iFeature = 0; iFeature < featureNum; iFeature++) {
		*mergedFeatures++ = *features++;
	}
	for(int iFeature = 0; iFeature < firstComponentSize; iFeature++) {
		*mergedFeatures++ = *firstComponent++;
	}
}

void traversePPT(int networkSize, int nodeId, int depth, int ppt[][3], int pptEnd, int nonLeavesEnd, 
	unsigned int **featureTwoArrays, int *featureNumOneArray, 
	unsigned int ***hashParametersThreeArrays, unsigned int **divisorsTwoArrays, int *parameterNum, unsigned int *fingerprint) {
	
	int currentNode;
	if(pptEnd == 0) {
		currentNode = ppt[0][0]; 
		minHash(hashParametersThreeArrays[0], divisorsTwoArrays[0], parameterNum[0], featureTwoArrays[currentNode], featureNumOneArray[currentNode], fingerprint);
		return;
	}
		
	unsigned int *features = NULL;
	unsigned int *mergedFeatures = NULL;
	int mergedFeatureNum = 0;
	int paraNum=0, featureNum=0;
	unsigned int* minHashValues = NULL;
	
	deque<struct ElementInQueue> *auxiliaryQueue = new deque<struct ElementInQueue>; 
	struct ElementInQueue elementInQueue;

	// for leaves
	unsigned int **lastHashParameters = NULL;
	unsigned int *lashDivisors = NULL;
	for(int iNode= pptEnd; iNode > nonLeavesEnd; iNode--) {
		currentNode = ppt[iNode][0]; 
		features = featureTwoArrays[currentNode]; 
		featureNum = featureNumOneArray[currentNode]; 
		
		lastHashParameters = hashParametersThreeArrays[depth]; 
		lashDivisors = divisorsTwoArrays[depth]; 
		paraNum = parameterNum[depth]; 
		
		minHashValues = new unsigned int [paraNum];
		minHash(lastHashParameters, lashDivisors, paraNum,features, featureNum, minHashValues);
		elementInQueue.minHashValues = minHashValues;
		elementInQueue.iNode = iNode;
		auxiliaryQueue->push_back(elementInQueue);
	}
	
	bool featureConcatenationFlag = false;
	struct ElementInQueue headElement;
	// for internal nodes
	for(int iNode = nonLeavesEnd; iNode >0; iNode--) {
		featureConcatenationFlag = false;
		currentNode = ppt[iNode][0];
		features = featureTwoArrays[currentNode];
		featureNum = featureNumOneArray[currentNode];
		paraNum = parameterNum[ppt[iNode][2]];   

		headElement = auxiliaryQueue->front();
		while(iNode == ppt[headElement.iNode][1]){
			auxiliaryQueue->pop_front();
			mergedFeatureNum = featureNum + parameterNum[ppt[headElement.iNode][2]];
			mergedFeatures = new unsigned int [mergedFeatureNum];
			mergeArrays(features, featureNum, headElement.minHashValues, parameterNum[ppt[headElement.iNode][2]], mergedFeatures);
		
			
			delete [](headElement.minHashValues);
			
			if(featureConcatenationFlag) {
				//delete features;
				delete[] features;
			}
			
			
			features = mergedFeatures;
			featureNum = mergedFeatureNum;
			
			if(!auxiliaryQueue->empty()) {
				headElement = auxiliaryQueue->front();
				featureConcatenationFlag = true;
			} else {
				break;
			}
			
		}
		minHashValues = new unsigned int [paraNum];
		minHash(hashParametersThreeArrays[ppt[iNode][2]], divisorsTwoArrays[ppt[iNode][2]], paraNum, mergedFeatures, mergedFeatureNum, minHashValues);
		elementInQueue.minHashValues = minHashValues;
		elementInQueue.iNode = iNode;
		auxiliaryQueue->push_back(elementInQueue);
		
		delete []mergedFeatures;
	}

	// for root
	featureConcatenationFlag = false;
	currentNode = ppt[0][0];
	features = featureTwoArrays[currentNode];
	featureNum = featureNumOneArray[currentNode];
	paraNum = parameterNum[0];
	while(!auxiliaryQueue->empty()) {
		headElement = auxiliaryQueue->front();
		auxiliaryQueue->pop_front();
		
		mergedFeatureNum = featureNum + parameterNum[ppt[headElement.iNode][2]];
		mergedFeatures = new unsigned int [mergedFeatureNum];
		mergeArrays(features, featureNum, headElement.minHashValues, parameterNum[ppt[headElement.iNode][2]], mergedFeatures);		
		delete [](headElement.minHashValues);
		
		if(featureConcatenationFlag) {
			//delete features;
			delete[] features;
		}
		featureConcatenationFlag = true;
		
		features = mergedFeatures;
		featureNum = mergedFeatureNum;
	}
	minHash(hashParametersThreeArrays[0], divisorsTwoArrays[0], paraNum, mergedFeatures, mergedFeatureNum, fingerprint);
	
	delete[] mergedFeatures;	
	delete auxiliaryQueue;
	
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
	float degreeEntropy = 0;
	int depth = 0;
	
	int i;
	if (argc == 1) {
		cout<<"NetHash"<<endl;
		cout<<"Options:"<<endl;
		cout<<"Parameters for algorithm:"<<endl;
		cout<<"-network <file>"<<endl;
		cout<<"Network data of adjlist from <file> for structure"<<endl;
		cout<<"-feature <file>"<<endl;
		cout<<"Feature data from <file> for contents"<<endl;
		cout<<"-hashdim <int>"<<endl;
		cout<<"Set dimension of network embeddings"<<endl;
		cout<<"-depth <int>"<<endl;
		cout<<"Depth of tree extracted from the network"<<endl;
		cout<<"-degreeentropy <float>"<<endl;
		cout<<"entropy of degrees"<<endl;	
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
	if ((i = ArgPos((char *)"-depth", argc, argv)) > 0) {
		depth = atoi(argv[i + 1]);
	}
	if ((i = ArgPos((char *)"-degreeentropy", argc, argv)) > 0) {
		degreeEntropy = atof(argv[i + 1]);
	}
	if ((i = ArgPos((char *)"-embedding", argc, argv)) > 0) {
		strcpy(embeddingFile, argv[i + 1]);
	}
	if ((i = ArgPos((char *)"-time", argc, argv)) > 0) {
		strcpy(timeFile, argv[i + 1]);
	}
	
	cout<< "hashdim: "<<k<< ", depth: "<< depth << ", degreeentropy: "<< degreeEntropy<<endl;
	
	// parameters for hash functions
	unsigned int* primeArray = new unsigned int[PRIME_NUM];
	readRandomPrimes("primes.txt", primeArray);
	unsigned int ***hashParametersThreeArrays = new unsigned int**[depth+1];	
	unsigned int **divisorsTwoArrays = new unsigned int*[depth+1];
	int *parameterNum = new int[depth+1];
	generateRandomPrimes(primeArray, k, degreeEntropy, depth, hashParametersThreeArrays, divisorsTwoArrays, parameterNum);	
	delete[] primeArray;
			
	// features on each node	
	unsigned int **tmpFeatureTwoArrays = new unsigned int*[MAX_NODE_NUM];
	int *tmpFeatureNumOneArray = new int[MAX_NODE_NUM];
	int networkSize = readFeatures(featureFile, tmpFeatureTwoArrays, tmpFeatureNumOneArray);	
	unsigned int **featureTwoArrays = new unsigned int*[networkSize];
	int *featureNumOneArray = new int[networkSize];
	for(int iNode = 0; iNode < networkSize; iNode++) {
		
		if(tmpFeatureNumOneArray[iNode] == 0) {
			featureTwoArrays[iNode] = NULL;
			featureNumOneArray[iNode] = 0;
		} else {
			featureTwoArrays[iNode] = new unsigned int[tmpFeatureNumOneArray[iNode]];
			for(int iFeature = 0; iFeature < tmpFeatureNumOneArray[iNode]; iFeature++) {
				featureTwoArrays[iNode][iFeature] = tmpFeatureTwoArrays[iNode][iFeature];
			}
			featureNumOneArray[iNode] = tmpFeatureNumOneArray[iNode];
		}
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

	int pptEnd;
	int nonLeavesEnd;
	unsigned int *fingerprint = NULL;
	struct PPTparameters pptParameters;
	clock_t timeBegin, timeEnd;
	double elapsedTime;
	unsigned int **fingerprints = new unsigned int *[networkSize];
	timeBegin = clock();	
	
	for(int iNode = 0; iNode < networkSize; iNode++) {
		pptParameters = buildParentPointerTree(network, adjNum, networkSize, iNode, depth, ppt);
		pptEnd = pptParameters.lengthOfPPT-1;
		nonLeavesEnd = pptEnd - pptParameters.numberOfLeaves;
		fingerprint = new unsigned int [k];
		traversePPT(networkSize, iNode, depth, ppt, pptEnd, nonLeavesEnd, featureTwoArrays, featureNumOneArray, hashParametersThreeArrays, divisorsTwoArrays, parameterNum, fingerprint);
		fingerprints[iNode] = fingerprint;		
	}
	timeEnd = clock();
	elapsedTime = ((double)(timeEnd - timeBegin))/CLOCKS_PER_SEC;
	cout<< elapsedTime<<endl;
	
	output(fingerprints, networkSize,  k, elapsedTime);
	
	return 0;
	
}
