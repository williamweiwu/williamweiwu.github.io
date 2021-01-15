clear all
clc

methods={'hashgnnplus'};
datasets = {'twitter', 'blog'};
turns = 5;
for i_data =6:length(datasets)
    data = datasets{i_data};
    for i_method = 1:length(methods)
        method = methods{i_method}; 
        
        for iteration = 1:5
        
            for dense = 0.5:0.1:0.9
                for turn = 1:turns
                    fingerprints = load([data, '/lp/', data, '.dense.', num2str(dense), '.', method, '.iteration.', num2str(iteration),  '.embedding.turn.', num2str(turn)]);
                    runtime = load([data, '/lp/', data, '.dense.', num2str(dense), '.', method, '.iteration.', num2str(iteration), '.time.turn.', num2str(turn)]);
                    
                    save([data, '/lp/', data, '.', num2str(dense), '.', method, '.fingerprints.iteration.', num2str(iteration), '.turn.', num2str(turn), '.mat'], 'fingerprints', 'runtime')
                end
            end
        end
        
    end
end
