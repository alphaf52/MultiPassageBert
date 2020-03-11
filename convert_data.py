
import pickle
import codecs
import torch

from read_ms_marco_data import InputFeatures 

input_files = ['data/small_train.dat'] # ['tmp.dat.0', 'tmp.dat.1', 'tmp.dat.2', 'tmp.dat.3', 'tmp.dat.4']
output_file = 'data/small_train_iter.dat' 

for input_file in input_files:
    c_features = torch.load(input_file)

    for (index, feature) in enumerate(c_features):
        if index % 1000 == 0:
            print(index)
        if not feature.is_impossible:
            with open(output_file, 'a') as fout:
                s = codecs.encode(pickle.dumps(feature), "base64").decode()
                fout.write(s + "\n")

    del c_features

