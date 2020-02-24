'''
Create word2vec embedding for the given sampling file(or the context file)
python Word2vec_temporal.py -f acm/_sampling_1_clique.csv
choose context file  either from 
1) sampling_baseline.csv   or
2) sampling_temporal_(0/1/2/3/4/5).csv 
Command: python Word2vec_temporal.py -f <context-file>

'''
import argparse, gensim
from gensim.models import word2vec
from argparse import RawTextHelpFormatter
import time

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

def generate_embeddings(context_file, sg, dimension, window, threads):
	try:
		context_file_ = open(context_file)
	except IOError:
		print('Could not read file: ', context_file)
		return

	print (context_file)
	contexts = word2vec.LineSentence(context_file)
	print('Generating word vectors...')
	model = word2vec.Word2Vec(contexts, sg=sg, size=dimension, window=window, workers=threads, min_count=1)
	print('Generated word vectors')
	context_file = context_file.split('_')[1]
	dataset = context_file.split('_')[0]
	embedding_file = context_file.replace(".csv","")
	embedding_file = embedding_file.replace("sampling","embedding")
	print("Embedding File: ", embedding_file)
	model_fname_bin =  'PPI-master/embeddings/embedding_file.bin' 
	model_fname_txt =  'PPI-master/embeddings/embedding_file.txt'
	print('Saving to ', model_fname_txt)
	model.save(model_fname_bin)
	model = gensim.models.Word2Vec.load(model_fname_bin)
	model.wv.save_word2vec_format(model_fname_txt, binary=False)
	print('Saved')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate author embeddings.', formatter_class=RawTextHelpFormatter)
	parser.add_argument('-f','--context_file', dest='context_file', required=True,
			    help='choose context file  either from 1) sampling_baseline.csv   or \n 2) sampling_temporal_(0/1/2/3/4/5).csv')
	parser.add_argument('-m','--model', dest='model', default=1,
			    help='Model used for generating vectors. \n0 - CBOW, 1 - skip-gram (default: 1)')
	parser.add_argument('-d','--dimension', dest='dimension', default=128,
			    help='Dimension of vectors generated by Word2Vec(default: 128)')
	parser.add_argument('-w','--window', dest='window_size', default=1,
			    help='Window size used for generating contexts(default: 1)')
	parser.add_argument('-t','--threads', dest='num_threads', default=30,
			    help='Number of threads used for generating contexts(default: 30)')
	args = parser.parse_args()

	start = time.time()

	generate_embeddings(context_file=args.context_file, sg=args.model, dimension=int(args.dimension), window=args.window_size, 
		threads=args.num_threads)

	finish = time.time()

	display_time_taken(finish - start)	