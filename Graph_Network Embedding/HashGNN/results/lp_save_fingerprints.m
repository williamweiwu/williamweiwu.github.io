clear all
clc

method='hashgnn';

datasets = {'twitter','facebook', 'blog' 'flickr',  'googleplus'};
iterations = [3,3,4,1,5];

turns = 5;
for i_data =1:length(datasets)
    data = datasets{i_data};
    iteration = iterations(i_data);

    for iturn =1:turns
        
        for dense = 0.5:0.1:0.9

            fingerprints = load([data, '/lp/', data, '.dense.', num2str(dense), '.', method, '.iteration.', num2str(iteration),  '.embedding.turn.', num2str(iturn)]);
            runtime = load([data, '/lp/', data, '.dense.', num2str(dense), '.', method, '.iteration.', num2str(iteration), '.time.turn.', num2str(iturn)]);
            
            save([data, '/lp/', data, '.', num2str(dense), '.', method, '.iteration.', num2str(iteration), '.fingerprints.turn.', num2str(iturn),'.mat'], 'fingerprints', 'runtime')
        end
    end
end


