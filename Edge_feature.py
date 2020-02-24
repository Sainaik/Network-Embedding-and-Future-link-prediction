'''
Compute hadamard product for all the author pair in test and random file
Thus we pick the mapping file for corresponding dataset and pick mapping from there

python compute_hadamard.py -d PPI-master -m temporal -r 1 -a average
'''
import os, csv
import shutil
import argparse
import multiprocessing as mp

from gensim.models import KeyedVectors

model = None


def compute_hadamard(authors):
	'''
	Given a author pair, compute hadamard 
	'''
	hadamard = model.wv[authors[0]] *  model.wv[authors[1]]
	return (authors[0], authors[1], hadamard)

def compute_average(authors):
	'''
	Given a author pair, compute average 
	'''
	average = (model.wv[authors[0]] +  model.wv[authors[1]])/2.0
	return (authors[0], authors[1], average)

def compute_l1(authors):
	'''
	Given a author pair, compute l1 
	'''
	l1 = abs(model.wv[authors[0]] -  model.wv[authors[1]])
	return (authors[0], authors[1], l1)

def compute_l2(authors):
	'''
	Given a author pair, compute l2 
	'''
	l2 = (abs(model.wv[authors[0]] -  model.wv[authors[1]]))**2
	return (authors[0], authors[1], l2)


def compute_similarities(dataset, method, num_threads):
	'''
	Read the embedding files from the embedding folder for given dataset and graphtype
	'''
	global model
	if method == "temporal":
		model_file = dataset + '/embeddings/embeddings_file.txt'
	elif method == "baseline":
		model_file = dataset + '/embeddings/embedding_baseline.txt'
	print("Reading the embedding file", model_file)
	model = KeyedVectors.load_word2vec_format(model_file)
	
	nodes_test = set()
	pool = mp.Pool(num_threads)

	files = ['train_edges' , 'train_ne', 'test_edges', 'random_edges']

	mapping_filename = dataset+'/mapping.csv'
	print("reading the mapping file", mapping_filename)
	mapping_reader = csv.reader(open(mapping_filename), delimiter=',')
	mapping_dict = {}

	for line in mapping_reader:
		mapping_dict[line[0]] = line[1]

	for test_file in files:
		edgelist = []
		reader = csv.reader(open(dataset+'/'+test_file+'.csv'), delimiter=',')
		for line in reader:
			node1 = mapping_dict[line[0]]
			node2 = mapping_dict[line[1]]
			nodes_test.add(node1)
			nodes_test.add(node2)
			
			if node1 not in model.wv.vocab:
				continue
			if node2 not in model.wv.vocab:
				continue
			
			edgelist.append((node1, node2))
		scores = ''
		if score == "hadamard":
			edgelist_scores = pool.map(compute_hadamard, edgelist)
		elif score == "average":
			edgelist_scores = pool.map(compute_average, edgelist)
		elif score == "l1":
			edgelist_scores = pool.map(compute_l1, edgelist)
		elif score == "l2":
			edgelist_scores = pool.map(compute_l2, edgelist)

		if method == "temporal":
			filename = dataset + '/hm_scores/' + test_file + '_temporal_' + str(kernel) + '_' + score + '.txt'
		elif method == "baseline":
			filename = dataset + '/hm_scores/' + test_file + '_baseline_' + score + '.txt'
		print("writing the scores file", filename)
		f = open(filename, 'w')
		for z in edgelist_scores:
			scores = z[0] + ' ' + z[1] 
			for item in z[2]:
				scores = scores + ' ' + str(item)
			f.write(scores)
			f.write('\n')
		f.close()
	   

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Compute hadamard operation on nodes participating to form an edge in test data.')
	parser.add_argument('-d','--dataset', dest='dataset', required=True,
			    help='Dataset PPI-master')
	parser.add_argument('-m','--method', dest='method', required=True,
			    help='The method used for generating embeddings. Choose among following: \ntemporal\n baseline')
	parser.add_argument('-t','--num_threads', dest='num_threads', default=30,
			    help='Number of threads used for parallelization(default: 30)')
	parser.add_argument('-r','--kernel', dest='kernel', default=0,
			    help='Recency kernel. 0: for linear 1-5: acc. to paper')
	parser.add_argument('-a','--score', dest='score', default="hadamard",
			    help='Score')

	args = parser.parse_args()
	global kernel, score
	kernel = int(args.kernel)
	score = args.score
	compute_similarities(args.dataset, args.method, int(args.num_threads))
