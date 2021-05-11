Introduction


WeightedMinHashToolbox provides the MinHash algorithm and the weighted MinHash algorithms in the paper titled "A Review for Weighted MinHash Algorithms". The paper elaborates the advantages and disadvantages of the algorithms and their application scenarios, and thus the users can select the appropriate methods in different scenes with the help of the paper. The algorithms aim to efficiently represent the weighted set as a compact fingerprint without the inefficient learning process. Based on the compact representation, the users can conduct many high-level data mining and machine learning tasks, e.g., classification, retrieval, clustering, visualization, etc.

If you use our toolbox, please cite the following paper:

@article{wu2018review,
  title={{A} {R}eview for {W}eighted {M}inHash {A}lgorithms},
  author={Wu, Wei and Li, Bin and Chen, Ling and Gao, Junbin and Zhang, Chengqi},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  year={2020},
  pages={1-1},
}

Before running the toolbox, please compile the following .cpp files because 4 algorithms call them. 

    g++ -std=c++11 cpluspluslib/gollapudi1_fingerprints.cpp -fPIC -shared -o cpluspluslib/gollapudi1_fingerprints.so

    g++ -std=c++11 cpluspluslib/haeupler_expandset.cpp -fPIC -shared -o cpluspluslib/haeupler_expandset.so

    g++ -std=c++11 cpluspluslib/cws_fingerprints.cpp -fPIC -shared -o cpluspluslib/cws_fingerprints.so

    g++ -std=c++11 cpluspluslib/haveliwala_expandset.cpp -fPIC -shared -o cpluspluslib/haveliwala_expandset.so
