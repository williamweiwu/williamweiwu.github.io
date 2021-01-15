clear all
clc

data = 'dblp11';

methods={'hashgnnplus'};
turns=5;
iterations=5;
sampledNodes = [1000 10000 100000 1000000];

for iSample = 1:length(sampledNodes)
    
    runtime_samples = zeros(turns, iterations);
    for turn =1:turns
        
        for iteration=1:iterations
            
            runtime = load([data, '/time.sample.', num2str(sampledNodes(iSample)),'.', method, '.iteration.', num2str(iteration), '.txt.turn.', num2str(turn)]);
            runtime_samples(turn, iteration) = runtime;
        end
        
    end
    runtimes = mean(runtime_samples, 1);
    save([data, '/', data, '.sample.', num2str(sampledNodes(iSample)),'.',method, '.scalability.runtime.mat'], 'runtimes')
end


runtime_original = zeros(turns, iterations);
for turn =1:turns
    
    for iteration=1:iterations
        
        runtime = load([data, '/time.',  method, '.iteration.', num2str(iteration), '.txt.turn.', num2str(turn)]);
        runtime_original(turn, iteration) = runtime;
    end
end
runtimes = mean(runtime_original, 1);

save([data, '/', data, '.',method, '.scalability.runtime.mat'], 'runtimes')