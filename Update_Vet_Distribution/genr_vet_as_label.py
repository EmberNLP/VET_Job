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

n_obs = 0
f = open("data/training_vet_as_label.txt", "w")
for i in range(m):
	vet_distribution = similarity_matrix[i]
	ind = np.argmax(vet_distribution)
	s = vet_distribution[ind]
	if s >= 0.5:
		f.write("__label__" + vet_category[ind].strip() + " " + ads_text[i].strip() + "\n")
		n_obs += 1

f.close()
print "Number of training observations: ", n_obs
# fasttext supervised -input data/training_vet_as_label.txt -output model/vet_as_label -lr 0.01 -epoch 40 -wordNgrams 1 -lrUpdateRate 1000 -loss hs 
# fasttext test model/job2vet.bin data/test_job.txt 
# fasttext print-sentence-vectors model/vet_as_label.bin < data/ads_text.txt > data/vectors_vet_as_label/ads_vectors.txt
# fasttext print-sentence-vectors model/vet_as_label.bin < data/vet_description.txt > data/vectors_vet_as_label/vet_vectors.txt



