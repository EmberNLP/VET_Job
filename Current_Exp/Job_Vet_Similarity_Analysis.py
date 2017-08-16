from collections import Counter
import operator
import numpy as np 
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity as cs


def read_txt(filename, filetype):
	l = []
	f = open(filename, "r")
	if filetype == "vectors":
		for line in f:
			l.append(line.strip().split())
	elif filetype == "text":
		for line in f:
			l.append(line.strip())
	return l

label_job = read_txt("data/label_job.txt", "text")

n = len(label_job)

label_job_frequency = Counter(label_job)
sorted_label_job_frequency = sorted(label_job_frequency.items(), key=operator.itemgetter(1), reverse = True)

filter_label_job_with_most_frequency = [label for label,count in sorted_label_job_frequency][:20]

# 'BusinessAndManagement', 'Banking,FinanceAndRelatedFields', 'Building', 'ElectricalAndElectronicEngineeringAndTechnology', 'SalesAndMarketing', 'FoodAndHospitality', 'ProcessAndResourcesEngineering', 'HumanWelfareStudiesAndServices', 'OfficeStudies', 'Nursing'

sample_size = 5000
ind_lists = []
for label in filter_label_job_with_most_frequency:
	ind_list = []
	for i in range(n):
		if label_job[i] == label:
			ind_list.append(i)
		if len(ind_list) == sample_size:
			break
	ind_lists.extend(ind_list)

ads_vectors = np.zeros((n,100))
# f = open("data/sentence_vectors_new_with_punctuation.txt")
f = open("vectors/ads_vectors.txt","r")
for i,line in enumerate(f):
	ads_vectors[i] = line.split()

ads = ads_vectors[ind_lists]

vet_label = read_txt("data/vet_category.txt", "text")
vet_vectors = read_txt("vectors/vet_vectors.txt", "vectors")
vets = np.array(vet_vectors, dtype = np.float)

similarity_matrix = cs(ads, vets)

mean_matrix = np.zeros((len(filter_label_job_with_most_frequency),len(vet_label)))
for i in range(len(vet_label)):
	c = similarity_matrix[:,i]
	for j in range(len(filter_label_job_with_most_frequency)):
		mean_matrix[j,i] = np.mean(c[j*sample_size:(j+1)*sample_size])

df = pd.DataFrame(mean_matrix.T, index = vet_label, columns = filter_label_job_with_most_frequency).to_csv("results/job_vet_similarity_result.csv")
