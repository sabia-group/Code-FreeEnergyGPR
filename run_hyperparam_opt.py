import numpy as np
from multiprocessing import Pool
from os import path

from fep_utils import list_of_directories, calc_mlh



dir_path = path.dirname(path.realpath(__file__)) + '/'

def _calculator_mlh(hp):
    mlh = calc_mlh(fpath_trainig, training_set, hp[0], hp[1], hp[2])
    return hp, mlh

def hp_optimisation(opt_sigma, opt_l, opt_se, stepm, limm):
    print(
            'sigma: '+str(opt_sigma)+
            '    l: '+str(opt_l)+
            '    se: '+str(opt_se)
            )
    if path.exists(fpath_trainig+'hp.npz'):
        hpload = np.load(fpath_trainig+'hp.npz')['hp']
    else:
        hpload = np.array([.1, 150, .1])
   
    hp_0 = hpload
    hp_d = np.array(hp_0)*stepm
    hp_lim = np.array(hp_0)*limm
    
    hp = []
    r_all = []
    while hp_d[0]>hp_lim[0] or hp_d[1]>hp_lim[1] or hp_d[2]>hp_lim[2]:
        hp_new = []
        for hp1 in [hp_0[0], hp_0[0]+hp_d[0], hp_0[0]-hp_d[0]]:
            for hp2 in [hp_0[1], hp_0[1]+hp_d[1], hp_0[1]-hp_d[1]]:
                for hp3 in [hp_0[2], hp_0[2]+hp_d[2], hp_0[2]-hp_d[2]]:
                    if not opt_sigma:
                        hp1=hpload[0]
                    if not opt_l:
                        hp2=hpload[1]
                    if not opt_se:
                        hp3=hpload[2]
                    if hp1>0 and hp2>1 and hp3>0.0001:
                        if (hp1, hp2, hp3) not in hp:
                            if (hp1, hp2, hp3) not in hp_new:
                                hp_new.append((hp1, hp2, hp3))
        if len(hp_new)>0:
            if __name__ == '__main__':
                p = Pool()
                r = p.map(_calculator_mlh, hp_new)
            hp = hp+hp_new
            r=np.array(r)
            p.close() 
            p.terminate()
            p.join()
            r_all = r_all + r.tolist()
            best = np.argmin(np.array(r_all)[:,1])
            hp_0 = r_all[best][0]
            print(
                    'sigma: '+str(r_all[best][0][0]) + 
                    '   l: '+str(r_all[best][0][1]) + 
                    '   se: '+str(r_all[best][0][2]) + 
                    '   mlh: '+str(r_all[best][1])
                    )
        else:
            hp_d = [h/2 for h in hp_d]
    np.savez(fpath_trainig+'hp.npz', hp=r_all[best][0])

fpath_trainig = dir_path+'training/'
training_set = list_of_directories(fpath_trainig)
hp_optimisation(True, False, False, 100, 10)
hp_optimisation(False, True, False, 10, 1)
hp_optimisation(False, False, True, 100, 10)

hp_optimisation(True, False, False, 10, 1)
hp_optimisation(False, True, False, 1, .1)
hp_optimisation(False, False, True, 10, 1)

hp_optimisation(True, False, False, 1, .1)
hp_optimisation(False, True, False, 1, .1)
hp_optimisation(False, False, True, 1, .1)

hp_optimisation(True, True, True, 1, .1)