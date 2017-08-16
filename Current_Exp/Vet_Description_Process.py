import os
import pandas as pd
import re

def clean_str(string): 

    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"this qualification is", "", string)
    string = re.sub(r"this qualification covers", "", string)
    string = re.sub(r"job roles and employment outcomes", "", string)
    string = re.sub(r"job roles", "", string)
    
    return string.strip()

name_list =[]
description_list = []

directory = "data/Descriptions"
for filename in os.listdir(directory):
	course_name = filename.split(".")[0]
	descriptions = []
	for line in open(directory + "/" + filename, "r"):
		line = line.strip()
		if len(line) != 0:
			descriptions.append(line)
	if len(descriptions) != 0:
		description_list.append(" ".join(clean_str(s.lower()) for s in descriptions))
		name_list.append(course_name)

df_description = pd.DataFrame({"VET_Code": name_list,
							   "Course_description": description_list})

VET_details = pd.read_csv("../data/Ad_VET/Datasets/VET_Details.csv")
df = pd.merge(df_description, VET_details[["VET_Code", "VET_Category"]], how='left', on=["VET_Code"])
df = df[["VET_Category","Course_description"]]
df.sort_values("VET_Category", inplace = True)
df = df.groupby(['VET_Category'])['Course_description'].apply(lambda x: " ".join(x)).reset_index()

df.to_csv("data/vet_description.csv", encoding = "UTF-8")

vet_list = list(df["VET_Category"])
vet_desription_list = list(df["Course_description"])

f_category = open("data/vet_category.txt", "w")
f_vet_description = open("data/vet_description.txt", "w")
for i in range(len(vet_list)):
	f_category.write(vet_list[i] + "\n")
	f_vet_description.write(vet_desription_list[i] + "\n")

f_category.close()
f_vet_description.close()


# fasttext supervised -input data/train_job.txt -output model/job2vet -lr 1.0 -epoch 40 -wordNgrams 2 -lrUpdateRate 1000 -loss hs -pretrainedVectors model/ads_w2v_100.vec
# fasttext test model/job2vet.bin data/test_job.txt 
# fasttext print-sentence-vectors model/job2vet.bin < data/ads_text.txt > data/ads_vectors.txt
# fasttext print-sentence-vectors model/job2vet.bin < data/vet_description.txt > data/vet_vectors.txt

