/******************************************

Copyright: Wei Wu

Author: Wei Wu

Description: This cpp file is used for "def haeupler(self, repeat=1, scale=1000)" in Weighted MinHash.py
                for the sake of efficiency from the algorithm, [Haeupler et. al., 2014].
                It preserves the remaining float part with probability after each weight is multiplied by a large constant.
                B. Haeupler, M. Manasse, and K. Talwar, "Consistent Weighted Sampling Made Fast, Small, and Easy",
                arXiv preprint arXiv: 1410.4266, 2014

Notes: compile the file,

    g++ -std=c++11 cpluspluslib/haeupler_expandset.cpp -fPIC -shared -o cpluspluslib/haeupler_expandset.so

******************************************/

# include <cmath>
# include <iostream>
# include "stdbool.h"
# include <random>


using namespace std;

/******************************************

Function: void ExpandSet(int, double *, int *, int, int, int, double *)

Description: Transform the weighted set, (feature_id, feature_weight), into the binary set, (expanded_feature_id)

Calls: GenerateExpandedSet(int, double *, int *, int, int, int, double *)

Parameters:
    expanded_set_predefined_size: the predefined size of the expanded set
    feature_weight: the array of the weights of the features
    feature_id: the array of the ids of the features
    feature_id_num: the number of the features whose weights are not zero in the data instance
    scale: a large constant to transform real-valued weights into integer ones
    expanded_feature_id: the binary set transformed from the original weighted set

Return: void

******************************************/

void ExpandSet(int expanded_set_predefined_size, double *feature_weight, int *feature_id, int feature_id_num, int scale, int seed, double *expanded_feature_id);

void ExpandSet(int expanded_set_predefined_size, double *feature_weight, int *feature_id, int feature_id_num, int scale, int seed, double *expanded_feature_id) {
 	int start_index = 0;
    int i_feature_weight = 0;

	for(int i_feature_id = 0; i_feature_id < feature_id_num; i_feature_id++) {
		for(i_feature_weight = 0; i_feature_weight < feature_weight[i_feature_id]; i_feature_weight++) {
			expanded_feature_id[start_index] = ((double)feature_id[i_feature_id] + 1) * scale * 10 + i_feature_weight;
			start_index ++;
		}

		std::mt19937_64 gen((unsigned long long)feature_id[i_feature_id] * pow(2, seed-1));
	    std::uniform_real_distribution<> dis(0.0, 1.0);

	    if(dis(gen) < feature_weight[i_feature_id] - floor(feature_weight[i_feature_id])) {
            expanded_feature_id[start_index] = ((double)feature_id[i_feature_id] + 1) * scale * 10 + i_feature_weight;
			start_index ++;
        }
	}
}

extern "C" {
    void GenerateExpandedSet(int expanded_set_predefined_size, double *feature_weight, int *feature_id, int feature_id_num, int scale, int seed, double *expanded_feature_id){
        ExpandSet(expanded_set_predefined_size, feature_weight, feature_id, feature_id_num, scale, seed, expanded_feature_id);
    }

}
