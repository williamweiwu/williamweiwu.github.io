#!/usr/bin/env python

import os


os.system("rm nethash");
os.system("g++ -std=c++11 -lm -O3 -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result nethash.cpp -o nethash -lgsl -lm -lgslcblas");
os.system("./nethash -network ./acm.adjlist -feature ./acm.feature -hashdim 200 -depth 2  -degreeentropy 3.26  -embedding ./acm.embedding -time ./acm.time");

