import numpy as np
from os import path

dir_path = path.dirname(path.realpath(__file__)) + '/'

#setting samples for prediction
fpath_prediction = dir_path+'prediction/'


file = open(fpath_prediction + 'fe.dat')
data = file.readlines()[1:]
file.close()

samples = [d.split()[0] for d in data]
fe = np.array([d.split()[1] for d in data], dtype=float)
fe = fe[np.argsort(samples)]


file = open(fpath_prediction + 'fep.dat')
data = file.readlines()[1:]
file.close()

samples = [d.split()[0] for d in data]
fep = np.array([d.split()[1] for d in data], dtype=float)
fep = fep[np.argsort(samples)]

samples = np.sort(samples)


for i in range(len(samples)):
    print(
            samples[i]+ ':\t(Fe) ' +str(fe[i])+ '\t\t(Fep) ' + str(fep[i]))