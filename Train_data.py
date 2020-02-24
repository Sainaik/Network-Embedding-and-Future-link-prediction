'''
Creates train data with pair on nodes which are present.
Steps:
1. Read all node pairs one by one.
2. Write the node pairs which are present before the threshold into the test data alnog with their date/month information.

Command: python 1train_data.py -d <dataset>
dataset PPI  

'''

import csv 
import pandas as pd
import argparse
import os


# threshold year to divide the dataset into train and test
threshold_year_start = 1970
threshold_year = 2013


# creating train dataset
def create_train(dataset):

	print('Creating train data (temporal) for ' + dataset + ' dataset')
	out_file = dataset+'/'+'PPI1_train_temporal.csv'
	f = open(out_file, 'w')

	# write edges
	reader = csv.reader(open(dataset+"/"+'PPI.csv'), delimiter=',')	
	#next(reader, None)  # skip the headers
	for row in reader:
		#print(row[0], row[1], row[2], row[3], row[4],row[5])
		[p1, p2, year, month,day] = [row[2], row[3], int(row[4]), row[5] , row[6]]
  
		if year <= threshold_year and year >= threshold_year_start:
			f.write(p1 + "," + p2 + "," + str(year) + "," + month +","+ day + "\n")
	f.close()
	print('Created train data (temporal) for ' + dataset + ' dataset: ' + out_file)
	print('Number of edges: ')
	os.system('wc -l ' + out_file)



if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Create training data')
	parser.add_argument('-d','--dataset', dest='dataset', required=True,
			    help='Dataset. Choose among following: \n PPI- master ')

	args = parser.parse_args()

	# create train data
	create_train(args.dataset)
