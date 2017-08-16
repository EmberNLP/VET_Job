import pandas as pd 
import numpy as np
from Utilize import *
from nltk.tokenize import word_tokenize
from collections import Counter
import operator

# Load data
Job_details = pd.read_csv("../data/Ad_VET/Datasets/Job_Details.csv")

# processing Job_details Dataset
print "Processing Job_details...."
Job_details = Job_details[["JOB_ID", "JobTitle", "Job_Category", "Description"]]
Job_details = Job_details.dropna(axis=0, how='any')
print "Number of Observations: ", len(Job_details.index)
print Job_details.columns.values

# filter categories
Job_Category_list = list(Job_details["Job_Category"])
category_frequency = Counter(Job_Category_list)
sorted_category_frequency = sorted(category_frequency.items(), key=operator.itemgetter(1), reverse = True)
# for item in sorted_category_frequency:
# 	print item
filtered_category = [c for c,count in sorted_category_frequency if count >= 650]
print len(filtered_category)

Job_details = Job_details[Job_details["Job_Category"].isin(filtered_category)]
print "Number of training Observations: ", len(Job_details.index)

# Generation with label: Job_Category    
train_file = open("data/train_job.txt","w")
text_ads = open("data/ads_text.txt","w")
label_ads = open("data/label_ads.txt","w")
for index,row in Job_details.iterrows():
	if (index + 1) % 1000 == 0:
		print index + 1
	try:
		__label = "__label__" + row["Job_Category"].replace(" ","")
 		__title = " ".join([ word_detection(word) for word in word_tokenize(clean_str(row["JobTitle"])) ])
 		__describe = " ".join([ word_detection(word) for word in word_tokenize(clean_str(row["Description"])) ])

	except:
		print row["JOB_ID"]
		continue

	train_file.write(__label + " " + __title + " " + __describe + "\n")  
	text_ads.write(__title + " " + __describe + "\n")  
	label_ads.write(__label[9:] + "\n")  

	# if index == 25000:
	# 	break

train_file.close() 
text_ads.close() 
label_ads.close() 

# fasttext supervised -input data/train_job.txt -output model/job2vet -lr 0.1 -epoch 40 -wordNgrams 2 -lrUpdateRate 1000 -loss hs 
# fasttext test model/job2vet.bin data/test_job.txt 
# fasttext print-sentence-vectors model/job2vet.bin < data/ads_text.txt > data/ads_vectors.txt
# fasttext print-sentence-vectors model/job2vet.bin < data/vet_description.txt > data/vet_vectors.txt



