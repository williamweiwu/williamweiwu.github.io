/******************************************

Copyright: Wei Wu

Author: Wei Wu

Description: This cpp file is used for "def cws(self, repeat=1)" in Weighted MinHash.py
                for the sake of efficiency from the Consistent Weighted Sampling (CWS) algorithm.
                It, as the first of the Consistent Weighted Sampling scheme, extends "active indices" from $[0, S]$
                in [Gollapudi et. al., 2006](1) to $[0, +\infty]$.
                M. Manasse, F. McSherry, and K. Talwar, "Consistent Weighted Sampling", Unpublished technical report, 2010.

Notes: compile the file,

    g++ -std=c++11 cpluspluslib/cws_fingerprints.cpp -fPIC -shared -o cpluspluslib/cws_fingerprints.so

******************************************/

# include <cmath>
# include <iostream>
# include "stdbool.h"
# include <random>
# include <climits>
# define PREDEFINED_ACTIVE_INDEX_NUM 1000

using namespace std;

double epsilon = 1E-4;

/******************************************

Function: void ActiveIndices(int , double , int , int , double *)

Description: Generate a sequence of "active indices", and return two special "active indices", $y_k$ and $z_k$, in (*yz)

Calls: GenerateFingerprint(int , double *, int *, int , int , double *, double *)

Parameters:
    feature_id: the id of the feature
    weight: the weight of the feature
    seed: random seed
    d_id: the id of the hash function, i.e., the index of the hash code
    yz: two special "active indices", $y_k$ and $z_k$

Return: void

******************************************/

void ActiveIndices(int feature_id, double weight, int seed, int d_id, double *yz);

/******************************************

Function: void GenerateFingerprint(int , double *, int *, int , int , double *, double *)

Description: Return the fingerprint for each instance, (fingerprint_k, fingerprint_y), in CWS

Calls: GenerateFingerprintOfInstance(int , double *, int *, int , int , double *, double *)

Parameters:
    dimension_num: the length of the hash code for each data instance
    feature_weight: the array of the weights of the features
    feature_id: the array of the ids of the features
    feature_id_num: the number of the features whose weights are not zero in the data instance
    seed: random seed
    fingerprint_k: one component of hash code $(k, y_k)$ for the data instance
    fingerprint_y: one component of hash code $(k, y_k)$ for the data instance

Return: void

******************************************/

void GenerateFingerprint(int dimension_num, double *feature_weight, int *feature_id, int feature_id_num, int seed, double *fingerprint_k, double *fingerprint_y);
/******************************************

Function: double Solve(double z, double beta)

Description: Solve the hash value of $y_k$ via binary search

Calls: GenerateFingerprint(int , double *, int *, int , int , double *, double *)

Parameters:
    z: a special "active index"
    beta: a uniform random number from $[0, 1]$

Return: The hash value of $y_k$

******************************************/

double Solve(double z, double beta);

void ActiveIndices(int feature_id, double weight, int seed, int d_id, double *yz) {
	int bound = ceil(log(weight)/log(2));

	std::mt19937 gen(feature_id * (d_id+1) * bound * pow(2, seed-1));
	std::uniform_real_distribution<double> uniform_dis(0.0, 1.0);
	double sample = pow(2, bound) * uniform_dis(gen);
	
	int i_sample = 0;
	int i_bound = 0;
	double samples[PREDEFINED_ACTIVE_INDEX_NUM] = {INT_MAX};
	samples[0] = sample;
	
	bool y_flag = false, z_flag  = false;
	while(sample > pow(2, bound-1)) {
		samples[i_sample] = sample;
		i_sample = i_sample +1;
		sample = sample * uniform_dis(gen);
	}

	for(int y_index = 0; y_index < PREDEFINED_ACTIVE_INDEX_NUM; y_index++) {
	    if(samples[y_index] <= weight) {
	        yz[0] = samples[y_index];
			y_flag = true;
			break;
	    }
	}
	while(!y_flag) {
		for(i_bound = bound-1; ; i_bound--) {

			std::mt19937 gen(feature_id * (d_id+1) * i_bound * pow(2, seed-1));
	        std::uniform_real_distribution<double> uniform_dis(0.0, 1.0);
	        sample = pow(2, i_bound) * uniform_dis(gen);
			if(sample > pow(2, i_bound-1)) {
				yz[0] = sample;
				y_flag = true;
				break;
			}
 		}
	}
	
	for(int z_index = i_sample-1; z_index >= 0; z_index--) {
		if(samples[z_index] > weight) {
			yz[1] = samples[z_index];
			z_flag = true;
			break;
		}
	}
	
	if(z_flag) {
		return;
	}
	
	i_bound = bound+1;
	while(1) {

		std::mt19937 gen(feature_id * (d_id+1) * i_bound * pow(2, seed-1));
	    std::uniform_real_distribution<double> uniform_dis(0.0, 1.0);
	    sample = pow(2, i_bound) * uniform_dis(gen);
		while(sample > pow(2, i_bound-1)) {
			yz[1] = sample;
			sample = sample * uniform_dis(gen);
			z_flag = true;
		}
		if(!z_flag) {
			i_bound = i_bound+1;
		} else {
			break;
		}
 	}
}

double Solve(double z, double beta) {
	
	double low = 0, high = 1, mid = 0;
	while(high-low > epsilon) {
		
		mid = (low+high)/2;
		if (abs(pow(mid,z)+pow(mid,z)*z*log(1/mid) - beta) < epsilon) {
			return mid;
		} else if(abs(pow(mid,z)+pow(mid,z)*z*log(1/mid) - beta) > epsilon) {
			high = mid;
		} else {
			low = mid;
		}
	}
}

void GenerateFingerprint(int dimension_num, double *feature_weight, int *feature_id, int feature_id_num, int seed, double *fingerprint_k, double *fingerprint_y) {

	double h_max = 0, beta = 0, h = 0;
	double yz[2];
	
	int d_id=0, i_nonzero = 0;
	for(d_id = 0; d_id < dimension_num; d_id++) {
		h_max = 0;

		for(i_nonzero = 0; i_nonzero < feature_id_num; i_nonzero++) {

			std::mt19937 gen(feature_id[i_nonzero] * (d_id+1) * pow(2, seed-1));
	        std::uniform_real_distribution<double> uniform_dis(0.0, 1.0);
	        beta = uniform_dis(gen);
			ActiveIndices(feature_id[i_nonzero], feature_weight[i_nonzero], seed, d_id, yz);
			h = Solve(yz[1], beta);
			if(h > h_max) {
				h_max = h;
				fingerprint_k[d_id] = feature_id[i_nonzero];
				fingerprint_y[d_id] = yz[0];
			}
		}
	}
}

extern "C" {
    void GenerateFingerprintOfInstance(int dimension_num, double *feature_weight, int *feature_id, int feature_id_num, int seed, double *fingerprint_k, double *fingerprint_y){
        GenerateFingerprint(dimension_num, feature_weight, feature_id, feature_id_num, seed, fingerprint_k, fingerprint_y);
    }
}

