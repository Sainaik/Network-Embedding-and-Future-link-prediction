import csv
import pandas as pd
from datetime import datetime
files= []
for i in range(1,7,1):
	files.append("PPI-master/Uncleaned/PPI_"+str(i)+".csv")

# output file path
f= open ("PPI-master/PPI.csv", 'w')
# read all datasets and get data into output
for fil in files:
	reader = csv.reader(open(fil),delimiter=',')
	next(reader, None)  # skip the headers
	for row in reader:
		[n1,n2,p1,p2,date]=[row[0],row[1],row[2],row[3],row[4]]
		m=date.split("/") 
		# m[-1] = years , m[0] = month, m[int(len(m)/2)] = day
		if(n1 !=' ' and n2 !=' '):
			f.write(n1+","+n2+","+p1+","+p2+","+m[-1]+","+m[0]+","+m[int(len(m)/2)]+"\n")
		

f.close()


