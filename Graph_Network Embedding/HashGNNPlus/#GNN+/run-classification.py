#!/usr/bin/env python

import os

dataset = ['m10','pubmed'];


method = 'hashgnnplus'
os.system("rm "+method);
os.system("g++ -std=c++11 -lm -O3 -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result "+method+".cpp -o "+method+" -lgsl -lm -lgslcblas");

iterations = [1,2,3,4,5]
turns = [1,2,3,4,5]



for data in dataset:
	print(data)

	path = "../../results/" + data + "/"
	folder = os.path.exists(path)

	if not folder:          
		os.makedirs(path) 

	for turn in turns:
		for iteration in iterations:
			os.system("./"+method+" -network ../../data/"+data+"/nethash/network.adjlist \
				-feature ../../data/"+data+"/nethash/features.txt -hashdim 200 -iteration "+str(iteration)+" \
				-embedding ../../results/"+data+"/"+data+"."+method+".iteration." + str(iteration)+".embeddings.turn." +str(turn) +\
				" -time ../../results/"+data+"/time."+method+".iteration." + str(iteration)+".txt.turn." +str(turn));

