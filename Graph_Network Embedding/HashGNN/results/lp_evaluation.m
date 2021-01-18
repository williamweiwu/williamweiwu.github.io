clear all

ratios = [0.5, 0.6, 0.7, 0.8, 0.9];

times = 10000;
method='hashgnn';
datasets = {'twitter','facebook', 'blog', 'flickr',  'googleplus'};
ks = [3, 3, 4, 1, 5];

iterations = 5;
turns =5;

for i_data =1:length(datasets)
    data = datasets{i_data};
    load(['../data/',data, '/',data, '.mat']);
    nodeNum = size(network,1);
    network(1:nodeNum + 1:end) = 0;
    
    auc = zeros(iterations, length(ratios), turns);
    runtimes = zeros(iterations, length(ratios), turns);
    elapsed= zeros(iterations, length(ratios), turns);
    cpu = zeros(iterations, length(ratios), turns);
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
        
        
        for dense= 1:length(ratios)
            
            k =ks(i_data);
            
            
            load(['./',data, '/lp/', data, '.', num2str(ratios(dense)), '.',method, '.iteration.', num2str(k), '.fingerprints.turn.', num2str(iturn),'.mat']);
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
            
            
            elapsed(k, dense, iturn) = cpu1+toc;
            auc(k, dense, iturn) = (greatNum + 0.5*equalNum)/ times;
            runtimes(k, dense, iturn) = runtime;
            cpu(k, dense, iturn) = elapsed(k, dense, iturn)+runtimes(k, dense, iturn);

        end
    end
    
    auc_mean = mean(auc, 3);
    cpu_mean = mean(cpu, 3);
    elapsed_mean = mean(elapsed, 3);
    runtimes_mean = mean(runtimes, 3);
    if ~exist(['./experiments'], 'dir')
        mkdir(['./experiments']);
    end
    save(['./experiments/', data, '.', method, '.results.mat'],  'auc', 'runtimes', 'elapsed','cpu', 'auc_mean', 'runtimes_mean', 'elapsed_mean','cpu_mean')
    
end









