clear all

ratios = [0.5, 0.6, 0.7, 0.8, 0.9];


times = 10000;

method='hashgnn';

datasets = {'twitter','facebook', 'blog' 'flickr',  'googleplus'};
turns = 5;
iterations = 5;
ks=100:50:300;

for i_data =1:length(datasets)
    data = datasets{i_data};
    load(['../data/',data, '/',data, '.mat']);
    nodeNum = size(network,1);
    network(1:nodeNum + 1:end) = 0;
    
        
        auc = zeros(length(ks),iterations, length(ratios),turns);
        runtimes = zeros(length(ks),iterations, length(ratios),turns);
        elapsed= zeros(length(ks),iterations, length(ratios),turns);
        cpu = zeros(length(ks),iterations, length(ratios),turns);
        
        for iturn =1:turns
            
            count = 1;
            nonexistence = zeros(2, times);
            tic;
            while count <= times
                edgeIds = randi(nodeNum, 2, 1);
                if network(edgeIds(1), edgeIds(2)) == 0
                    nonexistence(:, count) = edgeIds;
                    count = count+1;
                end
            end
            cpu1=toc;
  
            for ik = 1:length(ks)
                k = ks(ik);
                for dense= 1:length(ratios)

                    for iteration =1: iterations

                        load(['./',data, '/parameters/', data, '.', num2str(ratios(dense)), '.',method, '.iteration.', num2str(iteration), '.k.', num2str(k),'.fingerprints.turn.', num2str(iturn),'.mat']);
                        load(['../data/', data, '/', data, '_', num2str(ratios(dense)), '.mat'])

                        tic;
                        % train network
                        trainGraph(1:nodeNum + 1:end) = 0;
                        % hamming kernel is for minhash and its variation
                        nonexistence_similarity = sum(fingerprints(nonexistence(1,:),:)==fingerprints(nonexistence(2,:),:), 2)/size(fingerprints, 2);


                        % test network, missing links
                        testGraph(1:nodeNum + 1:end) = 0;
                        [iTest, jTest] = find(testGraph==1);
                        testedEdges = [iTest, jTest];
                        clear iTest
                        clear jTest
                        testedEdges = testedEdges(testedEdges(:,1)>testedEdges(:,2),:);
                        testedEdges = testedEdges(randi(size(testedEdges, 1), 1, times), :);

                        missing_similarity = sum(fingerprints(testedEdges(:,1),:)==fingerprints(testedEdges(:,2),:), 2)/size(fingerprints, 2);


                        % AUC
                        greatNum = sum(missing_similarity > nonexistence_similarity);
                        equalNum = sum(missing_similarity == nonexistence_similarity);


                        elapsed(ik, iteration, dense,iturn) = cpu1+toc;
                        auc(ik, iteration, dense,iturn) = (greatNum + 0.5*equalNum)/ times;
                        runtimes(ik, iteration, dense,iturn) = runtime;
                        cpu(ik, iteration, dense,iturn) = elapsed(ik, iteration, dense,iturn)+runtimes(ik, iteration, dense,iturn);

                    end
                end
            end
        end
        auc_mean = mean(auc, 4);
        cpu_mean = mean(cpu, 4);
        elapsed_mean = mean(elapsed, 4);
        runtimes_mean = mean(runtimes, 4);
        if ~exist(['./experiments'], 'dir')
            mkdir(['./experiments']);
        end
        save(['./experiments/', data, '.', method, '.parameters.results.mat'],  'auc', 'runtimes', 'elapsed','cpu', 'auc_mean', 'runtimes_mean', 'elapsed_mean','cpu_mean')

    
end









