import pandas as pd 
import numpy as np
from Utilize import *
from nltk.tokenize import word_tokenize
from collections import Counter
import operator


Job_Clean_Title_Description = pd.read_csv("../data/Ad_VET/Datasets/Job_Clean_Title_Description.csv")
Job_Details = pd.read_csv("../data/Ad_VET/Datasets/Job_Details.csv")
Job_VET_Keyword_Labels_Merged = pd.read_csv("../data/Ad_VET/Datasets/Job_VET_Keyword_Labels_Merged.csv")
VET_Details = pd.read_csv("../data/Ad_VET/Datasets/VET_Details.csv")

# Merge different datasets into one by keywords like JOB_ID
print "Number of Mapping: ", len(Job_VET_Keyword_Labels_Merged.index)
print "Number of Job Advertisements: ", len(Job_Details.index)
print "Merge Job Category..."
Model_data = pd.merge(Job_Clean_Title_Description, Job_Details[["JOB_ID", "Job_Category"]], how='left', on=["JOB_ID"])

print "Merge VET_Code..."
Model_data = pd.merge(Model_data, Job_VET_Keyword_Labels_Merged[["JOB_ID", "VET_Code"]], how='left', on=["JOB_ID"])

print "Merge VET_Category..."
Model_data = pd.merge(Model_data, VET_Details[["VET_Code", "VET_Category"]], how='left', on=["VET_Code"])
print "Column Names of merged Datasets: ", Model_data.columns.values
# ['JOB_ID' 'JobTitle' 'Description' 'Job_Category' 'VET_Code' 'VET_Category']

# Drop all the observations with any nan values
Model_data.drop("VET_Code", axis=1, inplace=True)
Model_data = Model_data.dropna(axis=0, how='any')
Model_data = Model_data.sort_values("JOB_ID")
Model_data = Model_data.drop_duplicates()

dataset = Model_data.groupby(["JOB_ID","JobTitle","Description","Job_Category"])["VET_Category"].apply(lambda x: "*".join(x)).reset_index()
print len(dataset.index)

# filter categories
Job_Category_list = list(dataset["Job_Category"])
category_frequency = Counter(Job_Category_list)
sorted_category_frequency = sorted(category_frequency.items(), key=operator.itemgetter(1), reverse = True)
# for item in sorted_category_frequency:
# 	print item
filtered_category = [c for c,count in sorted_category_frequency if count >= 200]
# print len(filtered_category)

dataset = dataset[dataset["Job_Category"].isin(filtered_category)]
print "Number of training Observations: ", len(dataset.index)

# Generation with label: Job_Category    
training_file = open("data/train_data.txt","w")
label_vet = open("data/label_vet.txt","w")
label_job = open("data/label_job.txt","w")
ads_text = open("data/ads_text.txt","w")
for index,row in dataset.iterrows():
	print index
	try:
		__label = "__label__" + row["Job_Category"].replace(" ","") + " "

 		title = " ".join([ word_detection(word) for word in word_tokenize(clean_str(row["JobTitle"])) ])
 		describe = " ".join([ word_detection(word) for word in word_tokenize(clean_str(row["Description"])) ])

		line =  __label + title + " " + describe

	except:
		print row["JOB_ID"]
		continue

	training_file.write(line + "\n")  
	label_vet.write(row["VET_Category"] + "\n")  
	label_job.write(row["Job_Category"] + "\n")  
	ads_text.write(title + " " + describe + "\n")  

training_file.close()
label_vet.close()
label_job.close()
ads_text.close()
