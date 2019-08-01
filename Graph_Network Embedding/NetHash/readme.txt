The folder contains the NetHash algorithm from IJCAI 2018, a large data set and primes used for hashing functions.
The algorithm embeds each node into the $l_1$ (hamming) space. 

If you use our algorithm and data sets in your research, please cite the following paper as reference in your publicaions:

Tip: if you have the issue of segmenation default when compilation, please modify the macro define, that is, #define PPT_SIZE, #define MAX_NODE_NUM 1000000000 and #define MAX_FEATURE_NUM 1000000, to the smaller values. 

@inproceedings{wu2018efficient,
  title     = {{E}fficient {A}ttributed {N}etwork {E}mbedding via {R}ecursive {R}andomized {H}ashing},
  author    = {Wei Wu and Bin Li and Ling Chen and Chengqi Zhang},
  booktitle = {IJCAI-18},          
  pages     = {2861--2867},
  year      = {2018}
}
