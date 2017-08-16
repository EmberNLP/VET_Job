import re

def clean_str(string): 

    string = re.sub(r"http\S+", "", string)
    string = re.sub(r"www\S+", "", string)
    string = re.sub(r"au", "", string)
    string = re.sub(r"com", "", string)
    string = re.sub(r"[^A-Za-z]", " ", string)

    return string.strip()

def word_detection(string):
	l = len(string)

	if l > 1:
		cut_index = [0]
		for i in range(1,l-1):
			if string[i].isupper():
				if string[i+1].islower() or string[i-1].islower():
					cut_index.append(i)
		if string[l-1].isupper() and string[l-2].islower():
			cut_index.append(l-1)
	else:
		return string.lower()

	if len(cut_index) == 1:
		return string.lower()
	else:
		output_string = []
		for i in range(1, len(cut_index)):
			output_string.append(string[ cut_index[i-1]:cut_index[i] ].lower())
		return " ".join(output_string) + " " + string[cut_index[-1]:].lower()

