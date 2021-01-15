#!/usr/bin/env python

import os

datasets = ['m10','pubmed'];

iterations = [1,2,3,4,5]
ks = [100,150,200,250,300]
turns = [1,2,3,4,5]

method = 'hashgnnplus'
os.system("rm "+method);
os.system("g++ -std=c++11 -lm -O3 -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result "+method+".cpp -o "+method+" -lgsl -lm -lgslcblas");

for turn in turns:
	for i_data, data in enumerate(datasets):
		#if i_data ==1:
		#	continue
		print(data)

		path = "../../results/parameters/" + data 
		folder = os.path.exists(path)
	 
		if not folder:          
			os.makedirs(path) 
	
		for k in ks:
			for iteration in iterations:
				os.system("./"+method+" -network ../../data/" + data + "/nethash/network.adjlist " + \
					" -feature ../../data/" + data + "/nethash/features.txt -hashdim " + str(k) + " -iteration " + str(iteration) + \
					" -embedding ../../results/parameters/" + data + "/" + data + "." +method + ".iteration." +str(iteration) + ".k."+ str(k) + ".embedding.turn." +str(turn) + \
					" -time ../../results/parameters/" + data + "/" + data + "."+ method+ ".iteration." + str(iteration) + ".k."+ str(k) + ".time.turn." +str(turn));

