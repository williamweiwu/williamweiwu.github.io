/******************************************

Copyright: Wei Wu

Author: Wei Wu

Description: This cpp file is used for "def gollapudi1(self, repeat=1, scale=1000)" in Weighted MinHash.py
                for the sake of efficiency from the algorithm, [Gollapudi et. al., 2006](1).
                It is an integer weighted MinHash algorithm, which skips much unnecessary hash value computation
                by employing the idea of "active index".
                S. Gollapudi and R. Panigraphy, "Exploiting Asymmetry in Hierarchical Topic Extraction",
                in CIKM, 2006, pp. 475-482.

Notes: compile the file,

    g++ -std=c++11 cpluspluslib/gollapudi1_fingerprints.cpp -fPIC -shared -o cpluspluslib/gollapudi1_fingerprints.so

******************************************/

# include <cmath>
# include <iostream>
# include "stdbool.h"
# include <random>

using namespace std;

/******************************************

Function: void GenerateFingerprint(int, double *, int *, int, int, double *, double *)

Description: Implement [Gollapudi et. al., 2006](1), and return the fingerprint for each data instance, (fingerprint_k, fingerprint_y)

Calls: GenerateFingerprintOfInstance(int, double *, int *, int, int, double *, double *)

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

void GenerateFingerprint(int dimension_num, double *feature_weight, int *feature_id, int feature_id_num, int seed, double *fingerprint_k, double *fingerprint_y) {

    double hash_value = 1;
    int skip = 0, i_block = 0, h_block  = 0;
    double min_hash = 1000;
    for(int d_id = 0; d_id < dimension_num; d_id++){
        min_hash = 1000;
        for(int i_nonzero = 0; i_nonzero < feature_id_num; i_nonzero++) {
            i_block = 0;

            std::mt19937_64 gen((unsigned long long)(feature_id[i_nonzero] * (d_id+1) * pow(2, seed-1)));
	        std::uniform_real_distribution<double> uniform_dis(0.0, 1.0);
	        hash_value = uniform_dis(gen);


            while (i_block < feature_weight[i_nonzero]) {
                std::geometric_distribution<int> geometric_dis(hash_value);
                skip = geometric_dis(gen) + 1;

                h_block = i_block;
                hash_value = hash_value * uniform_dis(gen);
                i_block = i_block+skip;
            }
            if(hash_value < min_hash) {
                fingerprint_k[d_id] = feature_id[i_nonzero];
                fingerprint_y[d_id] = h_block;
                min_hash = hash_value;
            }
        }
    }
}

extern "C" {
    void GenerateFingerprintOfInstance(int dimension_num, double *feature_weight, int *feature_id, int feature_id_num, int seed, double *fingerprint_k, double *fingerprint_y){
        GenerateFingerprint(dimension_num, feature_weight, feature_id, feature_id_num, seed, fingerprint_k, fingerprint_y);
    }
}
