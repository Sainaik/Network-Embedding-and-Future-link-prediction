'''
Generate corpus (create time-aware samples of the graph)
sigma: number of granules in which training data is divided as per time
r: recency kernel
num_samples: samples for each node
samples_length: length of each sample

example: python Corpus_gen.py -d PPI-master --samples_length 100 --num_samples 30 -sigma 86 -r 6
'''
import csv
import math
import time
#import randomed on python new p
import random 
import numpy as np
import networkx as nx
import multiprocessing as mp
import sys
import argparse
import matplotlib.pyplot as plt

threshold_year =  2013

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


def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
	    q[kk] = K*prob
	    if q[kk] < 1.0:
	        smaller.append(kk)
	    else:
	        larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
	    small = smaller.pop()
	    large = larger.pop()

	    J[small] = large
	    q[large] = q[large] + q[small] - 1.0
	    if q[large] < 1.0:
	        smaller.append(large)
	    else:
	        larger.append(large)

	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
	    return kk
	else:
	    return J[kk]

def get_alias_edge(src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		p = 1
		q = 1
		unnormalized_probs = []
		for dst_nbr in sorted(G.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
			elif G.has_edge(dst_nbr, src):
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			else:
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return alias_setup(normalized_probs)


def	create_graph(dataset, workers, sigma):
	'''
	Creates the newtworkx graph using the edgelist provided from train graph
	'''
	global G
	G = nx.Graph()
	mapping = {}
	cur_index = 0
	reader = csv.reader(open(dataset+'/PPI1_train_temporal.csv'), delimiter=',')
	#ww=open('Temporal_weights.csv','w')
	weights=[]
	Date=[]

	for row in reader:
		[edge1, edge2, year, month,day] = [row[0], row[1], int(row[2]), int(row[3]), int(row[4])]
		edge1_idx = -1
		edge2_idx = -1

		
		if edge1 not in mapping:
			mapping[edge1] = cur_index
			edge1_idx = cur_index
			cur_index = cur_index + 1
		else:
			edge1_idx = mapping[edge1]

		if edge2 not in mapping:
			mapping[edge2] = cur_index
			edge2_idx = cur_index
			cur_index = cur_index + 1
		else:
			edge2_idx = mapping[edge2]
		
		'''if dataset == "internet":
			w = float((threshold_year-year)*12 + (12-month))
		elif dataset == "fbwall":
			w = float((threshold_year-year-1)*12 + 1*11 + (12-month))
		else:
			w = float(threshold_year-year)'''
		w = ((threshold_year-year)*365+(12-month)*31+(1-day))
		if kernel == 0:
			w = 1.0/(w+1.0)
		elif kernel == 1:
			w = float(math.exp((-1.0*w*w)/(2.0*sigma*sigma)))
		elif kernel == 2:
			if sigma < w:
				w = 0.0
			else:
				w = 1.0 - (w/sigma)
		elif kernel == 3: 
			if sigma < w:
				w = 0.0
			else:
				w = 0.5*(1+math.cos((w*math.pi)/sigma))
		elif kernel == 4: 
			if sigma < w:
				w = 0.0
			else:
				w = math.sqrt(1-((w*w)/(sigma*sigma)))
		elif kernel == 5: 
			if sigma < w:
				w = 0.0
			else:
				w = 1.0
		elif kernel == 6: 		# baseline (unit weight)
			w = 1.0 
		weights.append(w)
		#ww.write(str(year)+','+str(month)+','+str(day)+','+str(w)+'\n')
		Date.append(year)#+'-'+str(month)+'-'+str(day))
		
		if G.has_edge(edge1_idx, edge2_idx):
			if kernel != 6:
				G[edge1_idx][edge2_idx]['weight'] = G[edge1_idx][edge2_idx]['weight']+w
		else:
			G.add_edge(edge1_idx, edge2_idx, weight=w)

	#ww.close()

	plt.plot(Date,weights, marker='o')

	plt.title('')

	plt.xlabel('TimeStamp')
	plt.ylabel('Time Decay Function Value')

	plt.show()
'''
	print(nx.info(G))

	mapping_filename = dataset+'/mapping.csv'
	f = open(mapping_filename, "w")
	for key, value in mapping.items():
		f.write("{0},{1}\n".format(key,value))
	f.close()
	print("mapping of node(a_) to integer index generated",mapping_filename)

	global alias_nodes, alias_edges

	alias_nodes = {}
	for node in G.nodes():
		unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
		alias_nodes[node] = alias_setup(normalized_probs)
	
	alias_edges = {}
	for edge in G.edges():
		alias_edges[edge] = get_alias_edge(edge[0], edge[1])
		alias_edges[(edge[1], edge[0])] = get_alias_edge(edge[1], edge[0])
	
	print ("aliasing done")


def func(node):
	#Find num_samples for the given node using khop random walk

	writer = []
	for walk_iter in range(num_samples):
		walk = [node]
		while len(walk) < samples_length:
			cur = walk[-1]
			cur_nbrs = sorted(G.neighbors(cur))
			if len(cur_nbrs) > 0:
				walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
			else:
				break
		s = ' '.join(str(i) for i in walk)
		writer.append(s)
	writ = '\n'.join(str(w) for w in writer )
	writ += "\n"
	return writ
		
def generate_corpus(dataset,workers):
    
	print("Generating corpus!")
	
	node_list = list(G.nodes())
	
	pool = mp.Pool(int(workers))

	if kernel == 6:
		sampling_filename = dataset+'/Corpus_baseline/sampling_baseline.csv'
	else:
		sampling_filename = dataset+'/Corpus_temporal/sampling_temporal_' + str(kernel)+"_Samples_"+str(num_samples)+ '.csv'

	print("sampling file will be generated: ",sampling_filename)

	write_file = pool.map(func, node_list)

	print ("Writing sampling to file ")

	f = open(sampling_filename,'w')
	
	for line in write_file:
		f.write(line)
	f.close()	
	print("Corpus generated")
'''



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Corpus(samples) Generation using train (positive) files')
	parser.add_argument('-d','--dataset', dest='dataset', required=True,
			    help='Dataset is PPI-master')
	parser.add_argument('-l','--samples_length', dest='samples_length', required=True,
			    help='length of samples')
	parser.add_argument('-n','--num_samples', dest='num_samples', required=True,
			    help='Number of samples')
	parser.add_argument('-r','--kernel', dest='kernel', required=True, default=0,
			    help='Recency kernel. 0: for linear 1-5: acc. to paper 6: baseline')
	parser.add_argument('-s','--sigma', dest='sigma', default=160100,
			    help='Recency sigma, equal to number of granules of time',action="store_true")
	parser.add_argument('-w','--workers', dest='workers', default=8,
			    help='number of cpu cores')
	args = parser.parse_args()

	global samples_length,num_samples,kernel
	samples_length = int(args.samples_length)
	num_samples = int(args.num_samples)
	kernel = int(args.kernel)
	sigma = float(args.sigma)
	start = time.time()

	create_graph(args.dataset, args.workers, sigma)
	#generate_corpus(args.dataset,args.workers)

	finish = time.time()

	display_time_taken(finish - start)	
	
