clear all
clc


datasets = {'m10', 'pubmed'};
methods={'hashgnnplus'};

turns = 5;
iterations=5;
for i_data =1:length(datasets)
    data = datasets{i_data};
    load(['../data/',data, '/',data, '.mat']);
    
    for i_method = 1:length(methods)
        method = methods{i_method};
        
        for iteration=1:iterations
            for turn =1:turns
            
                fingerprints = load([data, '/', data, '.', method, '.iteration.', num2str(iteration), '.embeddings.turn.', num2str(turn)]);
                runtime = load([data, '/time.', method, '.iteration.', num2str(iteration), '.txt.turn.', num2str(turn)]);

                save([data, '/', data, '.', method, '.fingerprints.iteration.', num2str(iteration), '.turn.', num2str(turn), '.mat'], 'fingerprints', 'runtime', '-v7.3')
            end
        end
    end
end
