import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity as cs
import pandas as pd 

# read data
def open_file(filename, type):
	f =  open(filename, "r")
	return_list = []
	for line in f:
		if type == "vector":
			return_list.append([float(item) for item in line.split()])
		elif type == "text":
			return_list.append(line.strip())
	return return_list

ads_vectors = open_file("data/ads_vectors.txt", "vector")
vet_vectors = open_file("data/vet_vectors.txt", "vector")
ads_text = open_file("data/ads_text.txt", "text")
vet_category = open_file("data/vet_category.txt", "text")

# compute similarity matrix
similarity_matrix = cs(np.array(ads_vectors), np.array(vet_vectors))
m,n = similarity_matrix.shape

# filter trust data index
filter_list = []
for i in range(n):
	s = similarity_matrix[:,i]
	_perc = np.percentile(s, 97.)
	_list = []
	for j in range(m):
		if s[j] >= _perc:
			_list.append(j)
	filter_list.append(set(_list))


vet_similarity_matrix = np.zeros(( len(vet_category), len(vet_category) ))
for i in range(len(vet_category)):
	for j in range(len(vet_category)):
		if i > j:
			vet_similarity_matrix[i,j] = len(filter_list[i].intersection(filter_list[j]))

vet_similarity_matrix /= len(_list)

pd.DataFrame(vet_similarity_matrix, index = vet_category, columns = vet_category).to_csv("vet_similarity.csv")
