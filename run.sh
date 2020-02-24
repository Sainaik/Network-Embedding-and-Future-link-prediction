#!/bin/sh

#python3 Train_data.py
#python3 Test_data.py 
#python3 Random_data.py
#python3 Train_ne.py -d PPI-master
python Corpus_Gen.py -d PPI-master -l 100 -n 30 -r 1 -w 5
python Word2Vec_temporal.py -f PPI-master/sampling_temporal_1.csv  -m 1 -d 128 -w 1 -t 10
python Edge_feature.py -d PPI-master -m temporal -t 10 -r 1 -a hadamard
python PPI_eval.py -d PPI-master -m temporal -r 1 -a hadamard
