from collections import Counter
import operator
import numpy as np 
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity as cs

f_label = open("data/label_ads.txt","r")

labels = []
for line in f_label:
	labels.append(line.strip())

f_label.close()

n = len(labels)
label_frequency = Counter(labels)
sorted_label_frequency = sorted(label_frequency.items(), key=operator.itemgetter(1), reverse = True)

filter_label_with_most_frequency = [label for label,count in sorted_label_frequency][:20]

# 'BusinessAndManagement', 'Banking,FinanceAndRelatedFields', 'Building', 'ElectricalAndElectronicEngineeringAndTechnology', 'SalesAndMarketing', 'FoodAndHospitality', 'ProcessAndResourcesEngineering', 'HumanWelfareStudiesAndServices', 'OfficeStudies', 'Nursing'

sample_size = 5000
ind_lists = []
for label in filter_label_with_most_frequency:
	ind_list = []
	for i in range(n):
		if labels[i] == label:
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
# similarity_matrix = cs(test_vectors, test_vectors)

# for i in range(10 - 1):
# 	print np.mean(similarity_matrix[170][ sample_size * i : sample_size * (i+1) ])

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

vet_label = read_txt("data/vet_category.txt", "text")
vet_vectors = read_txt("vectors/vet_vectors.txt", "vectors")
vets = np.array(vet_vectors, dtype = np.float)

similarity_matrix = cs(ads, vets)

mean_matrix = np.zeros((len(filter_label_with_most_frequency),len(vet_label)))
for i in range(len(vet_label)):
	c = similarity_matrix[:,i]
	for j in range(len(filter_label_with_most_frequency)):
		mean_matrix[j,i] = np.mean(c[j*sample_size:(j+1)*sample_size])

df = pd.DataFrame(mean_matrix.T, index = vet_label, columns = filter_label_with_most_frequency).to_csv("results/similarity_result.csv")
