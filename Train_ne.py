'''
Creates train data with pair on nodes which are NOT present.
Steps:
1. Read all node pairs one by one.
2. Write the node pairs which do not have any edge between them.

Command: python3 Train_ne.py -d PPI-master
'''

import csv
import math
import time
import random
import numpy as np
import networkx as nx
import multiprocessing as mp
from random import choice
import sys, os
import argparse

def display_time_taken(time_taken):
	'''
	time_taken: parameter to store the time taken
	the function for computing time taken
	'''
	time_str = ''
	if time_taken > 3600:
		hr = time_taken/3600
		time_taken = time_taken%3600
		time_str += str(int(hr)) + 'h '
	if time_taken > 60:
		mi = time_taken/60
		time_taken = time_taken%60
		time_str += str(int(mi)) + 'm '
	time_str += str(int(time_taken)) + 's'
	print('Time Taken: %s' % time_str)


def	create_graph(dataset):
	'''
	Creates the newtworkx graph using the edgelist provided from train graph
	'''
	G_train_e = nx.Graph()
	G_train_ne = nx.Graph()
	G_test_e = nx.Graph()
	G_test_ne = nx.Graph()

	reader = csv.reader(open(dataset+'/PPI1_train_temporal.csv'), delimiter=',')
	for row in reader:
		[edge1, edge2] = [row[0], row[1]]
		G_train_e.add_edge(edge1, edge2)
	reader = csv.reader(open(dataset+'/PPI2_test_temporal.csv'), delimiter=',')
	for row in reader:
		#print(row[1],row[0])
		[author1, author2] = [row[0], row[1]]
		G_test_e.add_edge(author1, author2)
	reader = csv.reader(open(dataset+'/random.csv'), delimiter=',')
	for row in reader:
		[author1, author2] = [row[0], row[1]]
		G_test_ne.add_edge(author1, author2)
	print("\ntrain_temporal data ")
	print(nx.info(G_train_e))
	print("\ntest data")
	print(nx.info(G_test_e))
	print("\nrandom data")
	print(nx.info(G_test_ne))

	edgelist_file = dataset+'/train_edges.csv'
	# nx.write_weighted_edgelist(G_train_e,edgelist_file)
	f = open(edgelist_file, 'w')
	for e in list(G_train_e.edges()):
		f.write(e[0] + "," + e[1] + "\n")
	f.close()
	print("edgelist generated ",edgelist_file)


	out_file = dataset+'/train_ne.csv'
	f = open(out_file, 'w')

	# get number of edges in positive train file
	g = open(edgelist_file, 'r')
	total = len(g.readlines())
	count = 0
	# find two random nodes from the above test graph, check if test and train files don't have that edgelist (i.e. negative edge)
	while(count < total):
		n1 = choice(list(G_train_e.nodes))
		n2 = choice(list(G_train_e.nodes))
		edges = dict()
		if not G_test_e.has_edge(n1, n2) and not G_test_ne.has_edge(n1, n2) and  not G_train_e.has_edge(n1,n2) and not (n1,n2) in edges and n1<n2:
				edges[(n1,n2)]={}
				f.write(n1 + "," + n2 + "\n")
				count += 1
	f.close()

	print("Created test random data for " + dataset + " dataset: " + out_file)
	print('Number of edges: ')
	os.system('wc -l ' + out_file) 



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Creates train data with pair on nodes which are NOT present.')
	parser.add_argument('-d','--dataset', dest='dataset', required=True,
			    help='Dataset is PPI-master')
	args = parser.parse_args()

	start = time.time()

	create_graph(args.dataset)

	finish = time.time()

	display_time_taken(finish - start)	
