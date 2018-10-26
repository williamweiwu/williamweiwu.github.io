Required packages:
numpy >= 1.15.0
scipy >= 1.1.0
ctypes >= 1.1.0
sympy >= 1.2


Before running the toolbox, please compile the following .cpp files because 4 algorithms call them. 
g++ -std=c++11 cpluspluslib/gollapudi1_fingerprints.cpp -fPIC -shared -o cpluspluslib/gollapudi1_fingerprints.so

g++ -std=c++11 cpluspluslib/haeupler_expandset.cpp -fPIC -shared -o cpluspluslib/haeupler_expandset.so

g++ -std=c++11 cpluspluslib/cws_fingerprints.cpp -fPIC -shared -o cpluspluslib/cws_fingerprints.so

g++ -std=c++11 cpluspluslib/haveliwala_expandset.cpp -fPIC -shared -o cpluspluslib/haveliwala_expandset.so

