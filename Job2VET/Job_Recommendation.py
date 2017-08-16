import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cs

def read_txt(filename, filetype):
	l = []
	f = open(filename, "r")
	if filetype == "vectors":
		for line in f:
			l.append(line.strip().split())
		return np.array(l, dtype = np.float)
	elif filetype == "text":
		for line in f:
			l.append(line.strip())
	return l

def top_k(array, K):
	return np.argpartition(array, -K)[-K:].tolist()


vet_label = read_txt("data/vet_category.txt", "text")
vet_vectors = read_txt("vectors/vet_vectors.txt", "vectors")
job_vectors = read_txt("vectors/ads_vectors.txt", "vectors")


similarity = cs(vet_vectors, job_vectors)

top_k_ind = []
top_k_value = []
for i in range(len(vet_label)):
	vet_s = similarity[i]
	ind = top_k(vet_s, 100)
	top_k_ind.append(ind)
	top_k_value.append(vet_s[ind].tolist())

# print top_k_ind[0]
# print top_k_value[0]
# print type(top_k_ind[0])
# print type(top_k_value[0])

f1 = open("top_k_jobs/top_k_ind.txt", "w")
f2 = open("top_k_jobs/top_k_value.txt", "w")

for i in range(len(top_k_ind)):
	f1.write(" ".join([ str(n) for n in top_k_ind[i] ]))
	f1.write("\n")
	f2.write(" ".join([ str(n) for n in top_k_value[i] ]))
	f2.write("\n")

f1.close()
f2.close()

def read_txt(filename):
	l = []
	f = open(filename, "r")
	for line in f:
		l.append(line.strip())
	return l


vet_label = read_txt("data/vet_category.txt")
top_k_ind = read_txt("top_k_jobs/top_k_ind.txt")
ads_text = read_txt("data/ads_text.txt")


for i in range(len(vet_label)):
	label = vet_label[i]
	f = open("top_k_jobs/" + label + "_jobs.txt","w")
	label_ind = [int(n) for n in top_k_ind[i].split()]
	for j in range(len(label_ind)):
		text = ads_text[label_ind[j]]
		f.write(text + "\n")
	f.close()