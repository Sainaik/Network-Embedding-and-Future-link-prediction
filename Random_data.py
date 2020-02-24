
import networkx as nx
import random
from random import choice
import csv
import argparse
import os
import matplotlib.pyplot as plt

# create random dataset for test (non-existing test edges)
def create_random(dataset):

	# create individual Author-Author graphs for train and test
	print("Creating individual Author-Author graphs for train and test")
	
	G_test = nx.Graph()
	G_train = nx.Graph()

	reader = csv.reader(open(dataset+'/PPI2_test_temporal.csv'), delimiter=',')
	for row in reader:
		[author1, author2] = [row[0], row[1]]
		G_test.add_edge(author1, author2)
	reader = csv.reader(open(dataset+'/train_raw.csv'), delimiter=',')
	for row in reader:
		[author1, author2] = [row[0], row[1]]
		G_train.add_edge(author1, author2)
	print("Graphs created")	
	#print(G_train.nodes())
	#print(G_test.edges())
	#nx.draw(G_train)
	#plt.show()
	#nx.draw(G_test)
	#plt.show()

	out_file = dataset+'/random_edges.csv'
	f = open(out_file, 'w')

	# get number of edges in positive (test file)
	count = 0
	# find two random nodes from the above test graph, check if test and train files don't have that edgelist (i.e. negative edge)
	total = len(G_test.edges())
	print (total)
	edges = dict()
	while(count < total):
		#n1 = choice(G_test.nodes())
		n1 = choice(list(G_test.nodes))
		n2 = choice(list(G_test.nodes))

		if not G_test.has_edge(n1, n2) and not G_train.has_edge(n1,n2) and not (n1,n2) in edges and n1<n2:
			edges[(n1,n2)]={}
			f.write(n1 + "," + n2 + "\n")
			count += 1
	f.close()

	print("Created test random data for " + dataset + " dataset: " + out_file)
	print('Number of edges: ')
	os.system('wc -l ' + out_file)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Create Random data for validation data')
	parser.add_argument('-d','--dataset', dest='dataset', required=True,
			    help='Dataset is PPI-master')

	args = parser.parse_args()

	dataset = args.dataset
	print("Creating test random data for " + dataset + " dataset")

	create_random(dataset)
	
