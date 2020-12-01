methods = { 'hashing_word'};
data = {'Australia', 'UK', 'America'};
keys = {'lockdown', 'border'};
temporal_weight = zeros(length(data), length(methods),length(keys), 108);
for i_data =1:length(data)
    for i_method =1:length(methods)

        for i_word =1:length(keys)

            load(['./',data{i_data},  '_', methods{i_method}, '.mat'])


            temporal_weight(i_data, i_method, i_word, 1:108) = keywords(i_word, 1:108);
        end
    end
end
