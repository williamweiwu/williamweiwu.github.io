#!/usr/bin/env python

import os



datasets = ['dblp11'];
sampledNodes = [1000, 10000, 100000,1000000]

iterations = [1, 2, 3, 4, 5];
turns = [1,2,3,4,5]
method = 'hashgnn'

for iteration in iterations:
    for data in datasets:
        for turn in turns:

            path = "../results/" + data + "/scalability/"
            folder = os.path.exists(path)

            if not folder:          
                os.makedirs(path) 

            for sampledNode in sampledNodes:

                os.system("./"+method+" -network ../data/"+data+"/"+data+"/network.sample."+str(sampledNode)+".adjlist \
                -feature ../data/"+data+"/"+data+"/features.sample."+str(sampledNode)+".txt \
                -hashdim 200 -iteration " + str(iteration) + " -embedding ../results/"+data+"/scalability/"+data+".sample."+str(sampledNode)+ "."+method+".iteration."+str(iteration) + ".embeddings.turn." +str(turn) +\
                " -time ../results/"+data+"/scalability/time.sample."+str(sampledNode)+"."+method+".iteration."+str(iteration)+ ".txt.turn." +str(turn));


            os.system("./"+method+" -network ../data/"+data+"/"+data+"/network.adjlist \
            -feature ../data/"+data+"/"+data+"/features.txt \
            -hashdim 200 -iteration " + str(iteration) + " -embedding ../results/"+data+"/scalability/"+data+"."+method+".iteration."+str(iteration) + ".embeddings.turn." +str(turn) +\
            " -time ../results/"+data+"/scalability/time."+method+".iteration."+str(iteration)+".txt.turn." +str(turn));

