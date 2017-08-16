# fasttext supervised -input data/train_job.txt -output model/job2vet -lr 0.1 -epoch 40 -wordNgrams 2 -lrUpdateRate 1000 -loss hs 
# fasttext test model/job2vet.bin data/test_job.txt 
fasttext print-sentence-vectors model/job2vet.bin < data/ads_text.txt > vectors/ads_vectors.txt
fasttext print-sentence-vectors model/job2vet.bin < data/vet_description.txt > vectors/vet_vectors.txt
