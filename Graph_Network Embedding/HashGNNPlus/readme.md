The datasets and source code of #GNN+ are for the paper, 
    #GNN+: Scalable Attributed Network Hashing, 
    an extended version of Hashing-Accelerated Graph Neural Networks for Link Prediction in WWW 2021.

The steps of running the repository in Linux:

Preparation work

    1. download and uncompress the package in the directory './HashGNNPlus'. 

    2. uncompress the dblp11 dataset
        'cd data/dblp11'
        'cat dblp11.tar.gz* | tar -zxv'

    3. confirm that the gcc library '-lgsl' has been installed. if no,
        'sudo apt-get install libgsl-dev'


Generation of node representation

    1. in the directory of '#GNN'
        'cd #GNN+/'

    2. compile the source code
        'python compile.py'

    3. run the node classification experiment
        'python run-classification.py'

    4. run the link prediction experiment
        'python run-prediction.py'

    5. run the scalability experiment
        'python run-scalability.py'

    6. run the parameter analysis experiment
        'python run-classification-parameters-analysis.py'
        'python run-classification-parameters-analysis.py'

Generation of experimental results in Matlab

    1. in the directory of 'results'

    2. node classification
        run hashgnnplus_classification_fingerprints.m
        run hasshgnnplus_classification.m
        the results are in '{data_name}/{data_name}.hashgnnplus.results.mat'
            micro_f1_mean, macro_f1_mean shown in Table 1
            cpus_mean shown in Table 3

    3. link prediction
        run hashgnnplus_prediction_fingerprints.m
        run hasshgnnplus_prediction.m
        the results are in '{data_name}/{data_name}.hashgnnplus.results.mat'
            auc_mean: AUC shown in Table 2
            cpu_mean: Runtime shown in Table 3

    4 scalability
        run scalability_runtime.m
        the results are in 'dblp11/dblp11.hashgnnplus.scalability.runtime.mat' (original DBLP) and 'dblp11/dblp11.sample.{node_number}.hashgnnplus.scalability.runtime.mat'

    5. parameter sensitivity
        run parameters_classification_fingerprints.m
        run parameters_classification.m
        the results are in 
        'parameters/{data_name}.hashgnnplus.results.mat'

        run parameters_prediction_fingerprints.m
        run parameters_prediction.m
        the results are in 
        'parameters/{data_name}.hashgnnplus.results.mat'
