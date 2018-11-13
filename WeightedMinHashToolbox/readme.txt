Required packages:
numpy >= 1.15.0
scipy >= 1.1.0
ctypes >= 1.1.0

Before running the toolbox, please compile the following .cpp files because 4 algorithms call them. 
g++ -std=c++11 cpluspluslib/gollapudi1_fingerprints.cpp -fPIC -shared -o cpluspluslib/gollapudi1_fingerprints.so

g++ -std=c++11 cpluspluslib/haeupler_expandset.cpp -fPIC -shared -o cpluspluslib/haeupler_expandset.so

g++ -std=c++11 cpluspluslib/cws_fingerprints.cpp -fPIC -shared -o cpluspluslib/cws_fingerprints.so

g++ -std=c++11 cpluspluslib/haveliwala_expandset.cpp -fPIC -shared -o cpluspluslib/haveliwala_expandset.so

The folder contains the toolbox for the MinHash algorithm and 12 weighted MinHash algorithms in the paper titled "A Review for Weighted MinHash Algorithms".
The algorithms transfer the weighted set into fingerprint. 

If you use our toolbox in your research, please cite the following paper as reference in your publicaions:

@inproceedings{wu2018review,
  title={{{A} {R}eview for {W}eighted {M}in{H}ash {A}lgorithms},
  author={Wu, Wei and Li, Bin and Chen, Ling and Gao, Junbin and Zhang, Chengqi},
  journal={arXiv preprint arXiv:1811.04633},
  year={2018}
}
