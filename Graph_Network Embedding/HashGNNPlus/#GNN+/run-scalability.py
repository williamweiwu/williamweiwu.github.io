#!/usr/bin/env python

import os


datasets = ['dblp11'];
sampledNodes = [1000, 10000, 100000, 1000000]

iterations = [1, 2, 3, 4, 5];
turns = [2,3,4,5]

method = 'hashgnnplus'
os.system("rm "+method);
os.system("g++ -std=c++11 -lm -O3 -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result "+method+".cpp -o "+method+" -lgsl -lm -lgslcblas");

for turn in turns:
    for iteration in iterations:
        for data in datasets:

            path = "../../results/" + data + "/"
            folder = os.path.exists(path)

            if not folder:          
                os.makedirs(path) 

            for sampledNode in sampledNodes:

                os.system("./"+method+" -network ../../data/"+data+"/"+data+"/nethash/network.sample."+str(sampledNode)+".adjlist \
                -feature ../../data/"+data+"/"+data+"/nethash/features.sample."+str(sampledNode)+".txt \
                -hashdim 200 -iteration " + str(iteration) + " -embedding ../../results/"+data+"/"+data+".sample."+str(sampledNode)+ "."+method+".iteration."+str(iteration) + ".embeddings.turn." +str(turn) +\
                " -time ../../results/"+data+"/time.sample."+str(sampledNode)+"."+method+".iteration."+str(iteration)+ ".txt.turn." +str(turn));


            os.system("./"+method+" -network ../../data/"+data+"/"+data+"/nethash/network.adjlist \
            -feature ../../data/"+data+"/"+data+"/nethash/features.txt \
            -hashdim 200 -iteration " + str(iteration) + " -embedding ../../results/"+data+"/"+data+"."+method+".iteration."+str(iteration) + ".embeddings.turn." +str(turn) +\
            " -time ../../results/"+data+"/time."+method+".iteration."+str(iteration)+".txt.turn." +str(turn));


