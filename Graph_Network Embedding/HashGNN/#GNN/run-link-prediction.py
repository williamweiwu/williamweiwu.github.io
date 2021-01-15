#!/usr/bin/env python

import os

denses = [0.5, 0.6, 0.7, 0.8, 0.9]
datasets = ['twitter','facebook', 'blog', 'flickr',  'googleplus'];
iterations = [3,3,4,1,5]
turns = [1,2,3,4,5]

method = 'hashgnn'

for i_data, data in enumerate(datasets):

	print(data)

	path = "../results/" + data + "/lp/"
	folder = os.path.exists(path)
 
	if not folder:          
		os.makedirs(path) 

	for turn in turns:
		for dense in denses:
			iteration = iterations[i_data]
			os.system("./"+method+" -network ../data/" + data + "/" + data + ".adjlist." + str(dense) + \
				" -feature ../data/" + data + "/features.txt -hashdim 200 -iteration " + str(iteration) + \
				" -embedding ../results/" + data + "/lp/" + data + ".dense." + str(dense) +"." +method + ".iteration." +str(iteration) + ".embedding.turn." +str(turn) + \
				" -time ../results/" + data + "/lp/" + data + ".dense." + str(dense) + "."+ method+ ".iteration." + str(iteration) + ".time.turn." +str(turn));

