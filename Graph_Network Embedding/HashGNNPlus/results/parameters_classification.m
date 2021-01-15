clear all
clc

datasets = {'m10', 'pubmed'};
methods={'hashgnnplus'};
turns = 5;
iterations=5;
trainRatios = [0.5 0.6 0.7 0.8 0.9];
ks=100:50:300;

for idata =1:length(datasets)
    
    data = datasets{idata};
    
    for i_method = 1:length(methods)
        method = methods{i_method};
        
        micro_f1 = zeros(length(ks),iterations, length(trainRatios), turns);
        macro_f1 = zeros(length(ks),iterations, length(trainRatios), turns);
        elapsed = zeros(length(ks),iterations, length(trainRatios), turns);
        runtimes = zeros(length(ks),iterations, turns);
        cpus = zeros(length(ks),iterations, length(trainRatios), turns);
        
        for ik = 1:length(ks)
            k = ks(ik);
            for turn =1:turns
                for iteration = 1:iterations
                    
                    load([ 'parameters/',data, '/',  data, '.', method, '.iteration.', num2str(iteration), '.k.', num2str(k), '.fingerprints.turn.', num2str(turn),'.mat'])
                    load (['../data/', data, '/', data, '.mat'])
                    runtimes(ik, iteration, turn) = runtime;
                    instanceNum = size(labels, 1);
                    
                    labelNum = length(unique(labels));
                    labelName = unique(labels);
                    
                    %rand('seed', iResult);
                    for iTrain = 1: length(trainRatios)
                        
                        
                        trainSamples = sort(randperm(instanceNum, round(instanceNum*trainRatios(iTrain))));
                        testSamples = setdiff(1:instanceNum, trainSamples);
                        
                        trainData = fingerprints(trainSamples,:);
                        trainLabels = labels(trainSamples,:);
                        
                        testData = fingerprints(testSamples,:);
                        testLabels = labels(testSamples,:);
                        
                        tic;
                        trainKernel = 1-squareform(pdist(trainData, 'hamming'));
                        model = svmtrain(trainLabels, [(1:length(trainSamples))', trainKernel], '-t 4 -q');
                        testKernel = 1-pdist2(testData, trainData, 'hamming');
                        [predictedLabels, accuracy, ~]= svmpredict(testLabels, [(1:length(testSamples))', testKernel], model, '-q');
                        elapsed(ik, iteration, iTrain, turn) = toc;
                        acc(turn, iTrain) = accuracy(1);
                        
                        tp = zeros(1, length(labelName));
                        fp = zeros(1, length(labelName));
                        fn = zeros(1, length(labelName));
                        for iLabel=1:length(labelName)
                            testPositive = find(testLabels == labelName(iLabel));
                            predictedPositive = find(predictedLabels == labelName(iLabel));
                            tp(iLabel) = length(intersect(predictedPositive, testPositive));
                            
                            testNegative = find(testLabels ~= labelName(iLabel));
                            fp(iLabel) = length(intersect(predictedPositive, testNegative));
                            
                            predictedNegative = find(predictedLabels ~= labelName(iLabel));
                            fn(iLabel) = length(intersect(predictedNegative, testPositive));
                            
                            %if isequal(length([testPositive; testNegative]), length(testLabels))==1
                            %    display('yes');
                            %end
                        end
                        
                        micro_f1(ik, iteration, iTrain, turn) = (2* nansum(tp))/(2*nansum(tp)+nansum(fp)+ nansum(fn));
                        macro_f1(ik, iteration, iTrain, turn) = (1/length(labelName)) * nansum((2*tp)./(2*tp+fp+fn));
                        cpus(ik, iteration, iTrain, turn) = elapsed(ik, iteration, iTrain, turn)+runtimes(ik, iteration, turn);
                    end
                end
            end
        end
        
        micro_f1_mean = mean(micro_f1, 4);
        micro_f1_std = std(micro_f1, 0, 4);
        macro_f1_mean = mean(macro_f1, 4);
        macro_f1_std = std(macro_f1, 0, 4);
        cpus_mean = mean(cpus, 4);
        cpus_std = std(cpus, 0, 4);
        elapsed_mean = mean(elapsed, 4);
        elapsed_std = std(elapsed, 0, 4);
        runtimes_mean = mean(runtimes,3);
        runtimes_std = std(runtimes, 0, 3);
        
        save([ 'parameters/',  data, '.', method, '.results.mat'], 'acc', 'micro_f1_mean', 'macro_f1_mean', 'runtimes_mean','cpus_mean', 'elapsed_mean', 'micro_f1_std', 'macro_f1_std', 'runtimes_std','cpus_std', 'elapsed_std')
    end
    
end
