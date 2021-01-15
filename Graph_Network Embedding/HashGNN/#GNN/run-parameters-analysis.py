#!/usr/bin/env python

import os

denses = [0.5, 0.6, 0.7, 0.8, 0.9]
datasets = ['twitter','facebook', 'blog', 'flickr',  'googleplus'];

iterations = [1,2,3,4,5]
ks = [100,150,200,250,300]
turns = [1,2,3,4,5]
method = 'hashgnn'

for turn in turns:
	for i_data, data in enumerate(datasets):

		print(data)

		path = "../results/" + data + "/parameters/"
		folder = os.path.exists(path)
	 
		if not folder:          
			os.makedirs(path) 
	
		for dense in denses:
			for k in ks:
				for iteration in iterations:
					os.system("./"+method+" -network ../data/" + data + "/" + data + ".adjlist." + str(dense) + \
						" -feature ../data/" + data + "/features.txt -hashdim " + str(k) + " -iteration " + str(iteration) + \
						" -embedding ../results/" + data + "/parameters/" + data + ".dense." + str(dense) +"." +method + ".iteration." +str(iteration) + ".k."+ str(k) + ".embedding.turn." +str(turn) + \
						" -time ../results/" + data + "/parameters/" + data + ".dense." + str(dense) + "."+ method+ ".iteration." + str(iteration) + ".k."+ str(k) + ".time.turn." +str(turn));

