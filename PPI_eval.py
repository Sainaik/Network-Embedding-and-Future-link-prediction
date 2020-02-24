'''
Pick the hadamard score(which is embedding for the author pair result) for various models
and do classification task using 
	lr
	dt
	rf
	nb

ex. python eval.py -d internet -model temporal -r 1 -a average
'''
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import os
import os.path
import argparse
import pandas as pd
import numpy as np
from sklearn import linear_model, tree
from sklearn import metrics
#from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import csv
from sklearn.svm import SVC


def eval(dataset, model):
	files = ['train_edges' , 'train_ne', 'test_edges', 'random_edges']
	if model == "temporal":
		train_exists = dataset + '/hm_scores/'+files[0]+'_temporal_'+ str(kernel)+"_"+str(score)+'.txt'
		train_non_exists = dataset + '/hm_scores/'+files[1] + '_temporal_' + str(kernel)+"_"+str(score)+'.txt'
		test_exists = dataset + '/hm_scores/' +files[2] + '_temporal_' + str(kernel)+"_"+str(score)+'.txt'
		test_non_exists = dataset + '/hm_scores/' +files[3] + '_temporal_' + str(kernel)+"_"+str(score)+'.txt'
	elif model == "baseline":
		train_exists = dataset + '/hm_scores/' +files[0] + '_baseline.txt'
		train_non_exists = dataset + '/hm_scores/'+files[1] + '_baseline.txt'
		test_exists = dataset + '/hm_scores/' + dataset + '_' +files[2] + '_baseline.txt'
		test_non_exists = dataset + '/hm_scores/'+files[3] + '_baseline.txt'
	print("Hello")
	print ("reading the embeddings file(hadamard scores)", train_exists)
	train_exist = csv.reader(open(train_exists), delimiter=' ')
	print ("reading the embeddings file(hadamard scores)", train_non_exists)
	train_non_exist = csv.reader(open(train_non_exists), delimiter=' ')
	print ("reading the embeddings file(hadamard scores)", test_exists)
	test_exist = csv.reader(open(test_exists), delimiter=' ')
	print ("reading the embeddings file(hadamard scores)", test_non_exists)
	test_non_exist = csv.reader(open(test_non_exists), delimiter=' ')
	print("saving for Kernal :",kernel,"Samples :",NoSamples, "Window size : ",window)
	
	data_header =list(range(1,129))  # for 100 features, since it discards last one
	data_header.append("Label")
	embeddings = []
	for lines in train_exist:
		data1 = lines[2:]
		data1.append(1)
		embeddings.append(data1)
		
	for lines in train_non_exist:
		data2 = lines[2:]
		data2.append(0)
		embeddings.append(data2)
	train_final = pd.DataFrame.from_records(embeddings, columns= data_header)
	#print(train_final)
	embedding = []
	for lines in test_exist:
		data3 = lines[2:]
		data3.append(1)
		embedding.append(data3)
		
	for lines in test_non_exist:
		data4 = lines[2:]
		data4.append(0)
		embedding.append(data4)

	test_final = pd.DataFrame.from_records(embedding, columns=data_header)

	result_final = []

	classifiers = ['lr', 'dt', 'rf', 'nb', 'svm']
	for classification_method in classifiers:
		print ("Training " + classification_method + " Classifier")
		train_x, train_y, test_x, test_y = train_final[data_header[:-1]], train_final[data_header[-1]],test_final[data_header[:-1]],test_final[data_header[-1]]
		
		if classification_method == 'lr':
			clf = linear_model.LogisticRegression()
			train_model = clf.fit(train_x, train_y)

		elif classification_method == 'dt':
			clf = tree.DecisionTreeClassifier()
			train_model = clf.fit(train_x, train_y)

		elif classification_method == 'rf':
			clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
			train_model = clf.fit(train_x, train_y)

		elif classification_method == 'nb':
			clf = GaussianNB()
			train_model = clf.fit(train_x, train_y)

		elif classification_method == 'svm':
			clf = SVC(gamma='auto')
			train_model = clf.fit(train_x, train_y)	

		mac_score = metrics.f1_score(test_y, train_model.predict(test_x), average = 'macro')
		mic_score = metrics.f1_score(test_y, train_model.predict(test_x), average = 'micro')
		ar_score = metrics.roc_auc_score(test_y, train_model.predict(test_x))
		avg_prec = metrics.average_precision_score(test_y, train_model.predict(test_x))
		acc = metrics.accuracy_score(test_y, train_model.predict(test_x), normalize=True)
		result_final.append([classification_method, mac_score, mic_score, ar_score, avg_prec, acc])
			
		Eval_Measure = ['Method', 'F-1_Macro', 'F-1_Micro', 'AUC', 'Avg_Precision', 'Accuracy' ]
		result_table = pd.DataFrame.from_records(result_final, columns=Eval_Measure)
		print (result_table)

	if model == "temporal":
		result_file_name = dataset+'/classification_results/'+dataset+ '_' + model + '_' + str(kernel) + '_'+ str(NoSamples) + '_' + str(window) + '_'+ score +'.txt'
	elif model == "baseline":
		result_file_name = dataset+'/classification_results/'+dataset+ '_' + model + '_' + str(NoSamples) + '_' + str(window) + score +'.txt'

	print("Writing Evaluation in file", result_file_name)	
	with open(result_file_name, 'w') as ff:
		ff.write(str(result_table))
		ff.write('/n') 

	
	
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Evaluate link prediction score using logistic regression classifier on edge features obtained using hadamard product.')
	parser.add_argument('-d','--dataset', dest='dataset', required=True,
			    help='Dataset is PPI-master')
	parser.add_argument('-m','--model', dest='model', required=True,
			    help='Choose model temporal')
	parser.add_argument('-r','--kernel', dest='kernel', default=1,
			    help='Recency kernel. 0: for linear 1-5: acc. to paper')
	parser.add_argument('-a','--score', dest='score', default="hadamard",
			    help='Score')
	#  we r using these to store the results uniqly for every sample and Window
	parser.add_argument('-n','--samples', dest='NoSamples',
			    help='No of samples used')
	parser.add_argument('-w','--window', dest='window',
			    help='window length used')	

	args = parser.parse_args()
	global kernel, score
	kernel = int(args.kernel)
	NoSamples=int(args.NoSamples)
	window=int(args.window)
	score = args.score
	eval(args.dataset, args.model)
