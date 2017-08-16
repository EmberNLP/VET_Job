import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity as cs

# read data
def open_file(filename, type):
	f =  open(filename, "r")
	return_list = []
	for line in f:
		if type == "vector":
			return_list.append([float(item) for item in line.split()])
		elif type == "text":
			return_list.append(line)
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
	filter_list.append(_list)


f = open("data/filtered_training_data.txt", "w")
for i,filter_index in enumerate(filter_list):
	_label = vet_category[i]
	print i, ": ", _label
	for ind in filter_index:
		f.write("__label__" + _label.strip() + " " + ads_text[ind].strip() + "\n")

f.close()

# fasttext supervised -input data/filtered_training_data.txt -output model/filtered_by_vet -lr 0.1 -epoch 40 -wordNgrams 1 -lrUpdateRate 1000 -loss hs 
# fasttext test model/job2vet.bin data/test_job.txt 
# fasttext print-sentence-vectors model/filtered_by_vet.bin < data/ads_text.txt > data/vectors_filtered_by_vet/ads_vectors.txt
# fasttext print-sentence-vectors model/filtered_by_vet.bin < data/vet_description.txt > data/vectors_filtered_by_vet/vet_vectors.txt



