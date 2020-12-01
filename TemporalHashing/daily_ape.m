methods = {'hashing'};
data = {'Australia', 'UK', 'America'};
temporal_ape = zeros(length(data), length(methods), 30);

for i_data =1:length(data)
    for i_method =1:length(methods)

        load(['./',data{i_data}, '_results.mat'])
        temporal_ape(i_data, i_method, :) = abs((mean(predicted_labels, 1)-double(testing_labels))./double(testing_labels));

    end
end

