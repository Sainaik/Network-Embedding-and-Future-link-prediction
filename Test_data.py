'''
Creates test data.
Steps:
1. Read the given mapped file row wise and based on the threshold decide if it is part of test (year > threshold_year).
2. Create edgelists for train and test (raw files) : Read the mapped file and write all edges in raw files, both for train and test data 
3. Remove edges from test_raw which are in train_raw
4. Create set of nodes present in train, store as embedded_nodes
5. Write edges to test where both nodes are in embedded_nodes

Command: python test_data.py -d <dataset>
'''

import csv 
import argparse
import os


# threshold year to divide the dataset into train and test
threshold_year_start = 1970
threshold_year = 2013

def create_raw(dataset):

	f = open( dataset + '/train_raw.csv', 'w')
	g = open(  dataset + '/test_raw.csv', 'w')
	
		# write all edges both for train and test data
	reader = csv.reader(open(dataset+'/PPI.csv'), delimiter=',')
	#next(reader, None)  # skip the headers
	for row in reader:
		[p1, p2, year,month,day] = [row[2], row[3],int(row[4]),row[5], row[6]]
		if year <= threshold_year and year >= threshold_year_start:
			f.write( p1 + "," +  p2 +"," +str(year) +","+ month+"," +day + "\n")
		elif year > threshold_year:
			g.write(p1 + "," +  p2 +"," +str(year) +","+ month+","+ day +"\n")
	f.close()
	g.close()
# create test dataset
def create_test(dataset):
	# remove edges from test_raw which are in train_raw
	
	command = 'sort ' + dataset + '/test_raw.csv ' +  dataset + '/train_raw.csv '+ dataset + '/train_raw.csv | uniq -u > ' + dataset+'/diff.csv'
	os.system(command)	
	embedded_nodes = set()
	reader = csv.reader(open(dataset+'/PPI.csv'), delimiter=',')
	#next(reader, None)  # skip the headers
	# print dataset 
	for row in reader:
		[p1, p2, year] = [row[2], row[3],int(row[4])]
		if year <= threshold_year and year >= threshold_year_start:
			embedded_nodes.add(p1)
			embedded_nodes.add(p2)

	# write edges to test where both nodes are in embedded_nodes
	out_file = dataset+'/test_edges.csv'
	f = open(dataset+'/diff.csv', 'r')
	g = open(out_file, 'w')

	reader = csv.reader(f, delimiter=',')
	#print(embedded_nodes)
	for row in reader:
		[p1, p2] = [row[0], row[1]]
		if p1 in embedded_nodes and p2 in embedded_nodes and p1<p2:
			g.write( p1 + "," +  p2 +"\n")

	
	f.close()
	g.close()
	command = 'rm ' + dataset+'/diff.csv'
	os.system(command)

	print("Created test data for " + dataset + " dataset: " + out_file)
	print('Number of edges: ')
	os.system('wc -l ' + out_file)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Creating test data')
	parser.add_argument('-d','--dataset', dest='dataset', required=True,
			    help='Here the Dataset is PPI-master')

	args = parser.parse_args()

	dataset = args.dataset

	print("Creating test data for " + dataset + " dataset")

	create_raw(dataset)
	create_test(dataset)




