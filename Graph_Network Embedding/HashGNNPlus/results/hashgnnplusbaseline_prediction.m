clear all

ratios = [0.5, 0.6, 0.7, 0.8, 0.9];

times = 10000;

methods={'hashgnnplusbaseline'};

datasets = {'twitter'};

turns =5;
iterations=5;

for i_data =1:length(datasets)

    data = datasets{i_data};
    load(['../data/',data, '/',data, '.mat']);
    nodeNum = size(network,1);
    network(1:nodeNum + 1:end) = 0;
    
    for i_method = 1:length(methods)
        method = methods{i_method};
        
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

        auc = zeros(iterations, length(ratios), turns);
        runtimes = zeros(iterations, length(ratios), turns);
        elapsed= zeros(iterations, length(ratios), turns);
        cpu = zeros(iterations, length(ratios), turns);
        
        for iTurn =1: 5
            for iteration = 1: iterations
                for dense= 1:5

                    for iturn=1:turns
                        load(['./',data, '/lp/', data, '.', num2str(ratios(dense)), '.',method, '.fingerprints.iteration.', num2str(iteration), '.turn.', num2str(iturn),'.mat']);
                        load(['../data/', data, '/lp/', data, '_', num2str(ratios(dense)), '.mat'])
                        
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
                        
                        
                        elapsed(iteration, dense, iturn) = cpu1+toc;
                        auc(iteration, dense, iturn) = (greatNum + 0.5*equalNum)/ times;
                        runtimes(iteration, dense, iturn) = runtime;
                        cpu(iteration, dense, iturn) = elapsed(iteration, dense, iturn)+runtimes(iteration, dense, iturn);
 
                    end
          
                end
            end
        end
        
        percentages = runtimes./cpu;
        percentages_mean = mean(percentages, 3);
        percentages_std = std(percentages, 0, 3);
        
        auc_mean = mean(auc, 3);
        auc_std = std(auc, 0, 3);
        cpu_mean = mean(cpu, 3);
        cpu_std = std(cpu, 0, 3);
        elapsed_mean = mean(elapsed, 3);
        elapsed_std = std(elapsed, 0, 3);
        runtimes_mean = mean(runtimes, 3);
        runtimes_std = std(runtimes, 0, 3);
        save(['./',data, '/', data, '.', method, '.results.mat'],  'auc_mean', 'cpu_mean', 'elapsed_mean', 'runtimes_mean', 'percentages_mean', 'percentages_std')
      
    end
end









