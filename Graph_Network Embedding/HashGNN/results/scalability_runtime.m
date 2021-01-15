clear all
clc

data = 'dblp11';


method='hashgnn';
turns=5;
iterations=5;
sampledNodes = [1000 10000 100000 1000000];

if ~exist(['./experiments'], 'dir')
    mkdir(['./experiments']);
end

for iSample = 1:length(sampledNodes)
    
    runtime_samples = zeros(turns, iterations);
    for turn =1:turns
        
        for iteration=1:iterations
            
            runtime = load([data, '/scalability/time.sample.', num2str(sampledNodes(iSample)),'.', method, '.iteration.', num2str(iteration), '.txt.turn.', num2str(turn)]);
            runtime_samples(turn, iteration) = runtime;
        end
        
    end
    runtimes = mean(runtime_samples, 1);
    save(['./experiments/', data, '.sample.', num2str(sampledNodes(iSample)),'.', method, '.scalability.runtime.mat'], 'runtimes')
end


runtime_original = zeros(turns, iterations);
for turn =1:turns
    
    for iteration=1:iterations
        
        runtime = load([data, '/scalability/time.',  method, '.iteration.', num2str(iteration), '.txt.turn.', num2str(turn)]);
        runtime_original(turn, iteration) = runtime;
    end
end
runtimes = mean(runtime_original, 1);

save(['./experiments/',  data, '.', method, '.scalability.runtime.mat'], 'runtimes')
