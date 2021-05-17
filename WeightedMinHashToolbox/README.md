**Introduction**


WeightedMinHashToolbox provides the MinHash algorithm and the weighted MinHash algorithms in the paper titled "A Review for Weighted MinHash Algorithms". The paper elaborates the advantages and disadvantages of the algorithms and their application scenarios, and thus the users can select the appropriate methods in different scenes with the help of the paper. The algorithms aim to efficiently represent the weighted set as a compact fingerprint without the inefficient learning process. Based on the compact representation, the users can conduct many high-level data mining and machine learning tasks, e.g., classification, retrieval, clustering, visualization, etc.


**Installation**

    pip install drhash
    
The homepage of the toolbox is [here](https://github.com/drhash-cn).


**Usage**

    # Input data: {array-like, sparse matrix}, shape (n_features, n_instances), format='csc'
    # a data matrix where row represents feature and column is data instance
    
    from drhash import WeightedMinHash
    from scipy import sparse

    data = sparse.rand(1000, 100, 0.2, format='csc')
    wmh = WeightedMinHash.WeightedMinHash(data, 50)
    fingerprints_k, fingerprints_y, elapsed = wmh.pcws()
    fingerprints, elapsed = wmh.licws()


**Citation**

If you use our toolbox, please cite the following paper:

@article{wu2018review,  
&emsp;&emsp;title={{A} {R}eview for {W}eighted {M}inHash {A}lgorithms},  
&emsp;&emsp;author={Wu, Wei and Li, Bin and Chen, Ling and Gao, Junbin and Zhang, Chengqi},  
&emsp;&emsp;journal={IEEE Transactions on Knowledge and Data Engineering},   
&emsp;&emsp;year={2020},  
&emsp;&emsp;pages={1-1},  
}


