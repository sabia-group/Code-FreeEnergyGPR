import numpy as np
from os import path

from fep_utils import list_of_directories, calc_alfa, calc_prediction

dir_path = path.dirname(path.realpath(__file__)) + '/'

#setting samples for prediction
fpath_training = dir_path+'training/'
training_set = list_of_directories(fpath_training)

hp = np.load(fpath_training+'hp.npz')['hp']
sigma = hp[0]
l = hp[1]
se = hp[2]
alfa, desc, K = calc_alfa(fpath_training, training_set, sigma, l, se)

fpath_prediction = dir_path+'prediction/'
prediction_set = list_of_directories(fpath_prediction)
Fp = calc_prediction(fpath_training, fpath_prediction, prediction_set, alfa, desc, K, sigma, l)

out_file = ['#sample_name	free_energy']
out_file = out_file + [ps + '    '+str(fp) for ps, fp in zip(prediction_set, Fp)]

file = open(fpath_prediction + 'fep.dat', 'w')
file.write('\n'.join(out_file))
file.close()