# include <cmath>
# include <iostream>
# include "stdlib.h"
# include "stdbool.h"
# include <random>

/*
g++ -std=c++11 cpluspluslib/haveliwala_expandset.cpp -fPIC -shared -o cpluspluslib/haveliwala_expandset.so
*/

using namespace std;

/******************************************

Function: void ExpandSet(int, double *, int *, int, int, double *)

Description: Transform the weighted set, (feature_id, feature_weight), into the binary set, (expanded_feature_id)

Calls: GenerateExpandedSet(int, double *, int *, int, int, double *)

Parameters:
    expanded_set_predefined_size: the predefined size of the expanded set
    feature_weight: the array of the weights of the features
    feature_id: the array of the ids of the features
    feature_id_num: the number of the features whose weights are not zero in the data instance
    scale: a large constant to transform real-valued weights into integer ones
    expanded_feature_id: the binary set transformed from the original weighted set

Return: void

******************************************/

void ExpandSet(int max_weight, double *feature_weight, int *feature_id, int feature_id_num, int scale, double *expanded_feature_id);

void ExpandSet(int max_weight, double *feature_weight, int *feature_id, int feature_id_num, int scale, double *expanded_feature_id) {
 	int start_index = 0;

	for(int i_feature_id = 0; i_feature_id < feature_id_num; i_feature_id++) {
		for(int i_feature_weight = 0; i_feature_weight < feature_weight[i_feature_id]; i_feature_weight++) {
			expanded_feature_id[start_index] = ((double)feature_id[i_feature_id]+1) * scale * 10  + i_feature_weight;
			start_index ++;
		}
	}
}

extern "C" {
    void GenerateExpandedSet(int max_weight, double *feature_weight, int *feature_id, int feature_id_num, int scale, double *expanded_feature_id){
        ExpandSet(max_weight, feature_weight, feature_id, feature_id_num, scale, expanded_feature_id);
    }

}

