/* ===================================================================================================== 
           					*** NetHash ***             
              			Author: Wei WU (william.third.wu@gmail.com)                
              			CAI, University of Technology Sydney (UTS)                 
 -------------------------------------------------------------------------------------------------------                              
 	Citation: W. Wu, B. Li, L. Chen, & C. Zhang, "Efficient Attributed Network Embedding via 
 			Recursive Randomized Hashing", IJCAI 2018.  					 
 ======================================================================================================= */


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
constexpr size_t  PPT_SIZE = 1000000000;
constexpr size_t  MAX_STRING = 255;
constexpr size_t  MAX_NODE_NUM = 1000000000;
constexpr size_t  MAX_FEATURE_NUM = 10000000;
constexpr size_t  MAX_PARAMETER_NUM = 300;
constexpr size_t  PRIME_NUM = 348513;
constexpr size_t  PPT_DIM2 = 3;

using namespace std;

/*
 Input: 
   network - adjacency list where each row represents a node and saves the indices of adjacency nodes
   feature - feature list where each row represents a node and saves the indices of features owned by the node
   hashdim - dimension of network embedding or fingerpints by hashing
   depth   - depth of rooted trees
   degreeentropy - entropy of degrees
 Output: 
   embedding - file which saves fingerprints for each node
   runtime - total runtime in seconds
*/

char networkFile[MAX_STRING], featureFile[MAX_STRING], embeddingFile[MAX_STRING], timeFile[MAX_STRING];

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
	size_t i=0;
	ifstream  fin(primesFile);
	if(!fin.is_open()) {
		cout<<"primes file not found!"<<endl;
		exit(1);
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
			for(size_t iNode=0; iNode < adjlist.size(); iNode++) {
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
		boost::split(featureVector, line, boost::is_any_of(" "), boost::token_compress_on);
	
		featureArray = new unsigned int[featureVector.size()];
		for(size_t iFeature=0; iFeature < featureVector.size(); iFeature++) {
			featureArray[iFeature]= stoi(featureVector.at(iFeature));
		}
		tmpFeatureTwoArrays[nodeNum] = featureArray;
		tmpFeatureNumOneArray[nodeNum] = featureVector.size();
		nodeNum++;
		featureVector.clear();
	}
	fin.close();

	return nodeNum;
}


struct PPTparameters buildParentPointerTree(int **network, int *adjNum, int networkSize, int nodeId, int depth, int ppt[]) {
		
	struct PPTparameters pptParameters;
	
	if(!network[nodeId]) {
		pptParameters.lengthOfPPT = 1;
		pptParameters.numberOfLeaves = 1;
		return pptParameters;
	}
	
	// node as root of a ppt
	int pptSize = 1;
	ppt[0] = nodeId;
	ppt[1] = -1;
	ppt[2] = 0;

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
			currentNode = ppt[iNode*PPT_DIM2];
			children = network[currentNode];
			for(int iChildren=0; iChildren < adjNum[currentNode]; iChildren++) {
				ppt[pptSize*PPT_DIM2] = children[iChildren];
				ppt[pptSize*PPT_DIM2 + 1] = iParent;
				ppt[pptSize*PPT_DIM2 + 2] = d;
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
	static unsigned  *result = new unsigned[MAX_FEATURE_NUM*MAX_PARAMETER_NUM];  // Note: released on app completion
	//unsigned int  result[MAX_FEATURE_NUM][MAX_PARAMETER_NUM];
	for(int iRow = 0; iRow< featureNum; iRow++) {	
		for(int jColumn = 0; jColumn < parameterNum; jColumn++) {
			result[iRow*MAX_PARAMETER_NUM + jColumn] = (features[iRow]* hashParameters[jColumn][0] + hashParameters[jColumn][1]) % divisors[jColumn];
		}
	}
		
	unsigned int minValue;
	int minIndex;
	for(int iColumn = 0; iColumn < parameterNum; iColumn++) {	
		minValue = result[iColumn];
		minIndex = 0;
		for(int jRow = 1; jRow < featureNum; jRow++) {
			if(minValue > result[jRow*MAX_PARAMETER_NUM + iColumn]) {
				minValue = result[jRow*MAX_PARAMETER_NUM + iColumn];
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

void traversePPT(int networkSize, int nodeId, int depth, int ppt[], int pptEnd, int nonLeavesEnd, 
	unsigned int **featureTwoArrays, int *featureNumOneArray, 
	unsigned int ***hashParametersThreeArrays, unsigned int **divisorsTwoArrays, int *parameterNum, unsigned int *fingerprint) {
	
	int currentNode;
	
	if(pptEnd == 0) {
		currentNode = ppt[0]; 
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
		currentNode = ppt[iNode*PPT_DIM2]; 
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
		currentNode = ppt[iNode*PPT_DIM2];
		features = featureTwoArrays[currentNode];
		featureNum = featureNumOneArray[currentNode];
		paraNum = parameterNum[ppt[iNode*PPT_DIM2 + 2]];   

		headElement = auxiliaryQueue->front();
		while(iNode == ppt[headElement.iNode*PPT_DIM2 + 1]){

			auxiliaryQueue->pop_front();
			mergedFeatureNum = featureNum + parameterNum[ppt[headElement.iNode*PPT_DIM2 + 2]];
			mergedFeatures = new unsigned int [mergedFeatureNum];
			mergeArrays(features, featureNum, headElement.minHashValues, parameterNum[ppt[headElement.iNode*PPT_DIM2 + 2]], mergedFeatures);
		
			
			delete [](headElement.minHashValues);
			
			if(featureConcatenationFlag) {
				delete []features;
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
		minHash(hashParametersThreeArrays[ppt[iNode*PPT_DIM2 + 2]], divisorsTwoArrays[ppt[iNode*PPT_DIM2 + 2]], paraNum, mergedFeatures, mergedFeatureNum, minHashValues);

		elementInQueue.minHashValues = minHashValues;
		elementInQueue.iNode = iNode;
		auxiliaryQueue->push_back(elementInQueue);
		
		delete []mergedFeatures;
	}

	// for root
	featureConcatenationFlag = false;
	currentNode = ppt[0];
	features = featureTwoArrays[currentNode];
	featureNum = featureNumOneArray[currentNode];
	paraNum = parameterNum[0];
	while(!auxiliaryQueue->empty()) {
		headElement = auxiliaryQueue->front();
		auxiliaryQueue->pop_front();
		
		mergedFeatureNum = featureNum + parameterNum[ppt[headElement.iNode*PPT_DIM2 + 2]];
		mergedFeatures = new unsigned int [mergedFeatureNum];
		mergeArrays(features, featureNum, headElement.minHashValues, parameterNum[ppt[headElement.iNode*PPT_DIM2 + 2]], mergedFeatures);		
		delete [](headElement.minHashValues);
		
		if(featureConcatenationFlag) {
			delete []features;
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

int main(int argc, char* argv[]) {
	
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
		featureTwoArrays[iNode] = new unsigned int[tmpFeatureNumOneArray[iNode]];
		for(int iFeature = 0; iFeature < tmpFeatureNumOneArray[iNode]; iFeature++) {
			featureTwoArrays[iNode][iFeature] = tmpFeatureTwoArrays[iNode][iFeature];
		}
		featureNumOneArray[iNode] = tmpFeatureNumOneArray[iNode];
	}
	//delete temporary memory
	for(size_t iNode = 0; iNode < MAX_NODE_NUM; iNode++) {
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
	timeBegin = time(NULL);	
	int *ppt = new int[PPT_SIZE*PPT_DIM2];  // [PPT_SIZE][3]
	
	for(int iNode = 0; iNode < networkSize; iNode++) {
		pptParameters = buildParentPointerTree(network, adjNum, networkSize, iNode, depth, ppt);
		pptEnd = pptParameters.lengthOfPPT-1;
		nonLeavesEnd = pptEnd - pptParameters.numberOfLeaves;
		fingerprint = new unsigned int [k];
		traversePPT(networkSize, iNode, depth, ppt, pptEnd, nonLeavesEnd, featureTwoArrays, featureNumOneArray, hashParametersThreeArrays, divisorsTwoArrays, parameterNum, fingerprint);
		fingerprints[iNode] = fingerprint;		
	}
	delete[] ppt;  // Note: ppt is anyway automatically released on the app completion
	timeEnd = time(NULL);
	elapsedTime = timeEnd - timeBegin;
	cout<< "elapsed time: "<< elapsedTime << endl;
	
	output(fingerprints, networkSize,  k, elapsedTime);
	
	return 0;
	
}
