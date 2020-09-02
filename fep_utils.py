from os import listdir, path
import numpy as np

mases_to_symbols = {1.0080:'H', 12.0107:'C', 15.9994:'O'}

def list_of_files(fpath):
    return [f for f in listdir(fpath) if path.isfile(path.join(fpath, f))]

def list_of_directories(fpath):
    return list(set(listdir(fpath))-set(list_of_files(fpath)))

def angle(a,b):
    r = np.dot(a, b)/(np.linalg.norm(a, axis=0)*np.linalg.norm(b, axis=0))
    return np.arccos(r)

def u_rotate(phi, u):
    norm_u = np.power(np.sum(np.power(u,2)),.5)
    u = u/norm_u
    rot_mat = np.zeros((3, 3))
    
    rot_mat[0,0] = np.cos(phi)+np.power(u[0],2)*(1-np.cos(phi))
    rot_mat[0,1] = u[0]*u[1]*(1-np.cos(phi))-u[2]*np.sin(phi)
    rot_mat[0,2] = u[0]*u[2]*(1-np.cos(phi))+u[1]*np.sin(phi)
    
    rot_mat[1,0] = u[0]*u[1]*(1-np.cos(phi))+u[2]*np.sin(phi)
    rot_mat[1,1] = np.cos(phi)+np.power(u[1],2)*(1-np.cos(phi))
    rot_mat[1,2] = u[2]*u[1]*(1-np.cos(phi))-u[0]*np.sin(phi)
    
    rot_mat[2,0] = u[0]*u[2]*(1-np.cos(phi))-u[1]*np.sin(phi)
    rot_mat[2,1] = u[2]*u[1]*(1-np.cos(phi))+u[0]*np.sin(phi)
    rot_mat[2,2] = np.cos(phi)+np.power(u[2],2)*(1-np.cos(phi))

def _parsing_poscar(lines):
    name = lines[0]
    lattice_cons = float(lines[1])
    basex = np.array([float(l)*lattice_cons for l in lines[2].split()])
    basey = np.array([float(l)*lattice_cons for l in lines[3].split()])
    basez = np.array([float(l)*lattice_cons for l in lines[4].split()])
    atoms = [int(l) for l in lines[6].split()]
    atoms_tot = sum(atoms)
    header_tot = 8
    positions = lines[header_tot:]
    positions = [p[:-1] for p in positions]
    positions = np.array([p.split() for p in positions], float)
    return name, basex, basey, basez, atoms_tot, positions
    
def _rotating_positions(basex, basey, basez, positions):
    #fixing x    
    if basex[1] != 0 or basex[2] != 0:
        rotation_vecotr = np.cross([1,0,0], basex)
        rotation_angle = angle([1,0,0], basex)
        rotation_matrix = u_rotate(rotation_angle, rotation_vecotr)
        basez = np.dot(basez, rotation_matrix)
        basey = np.dot(basey, rotation_matrix)
        basex = np.dot(basex, rotation_matrix)
        for i in range(len(positions)):
            positions[i, :] = np.dot(positions[i,:], rotation_matrix)

    #fixing y
    if basey[2] != 0:
        rotation_vecotr = [1,0,0]
        rotation_angle = -angle([0,1,0], [0, basey[1], basey[2]])
        rotation_matrix = u_rotate(rotation_angle, rotation_vecotr)
        basez = np.dot(basez, rotation_matrix)
        basey = np.dot(basey, rotation_matrix)
        basex = np.dot(basex, rotation_matrix)
        for i in range(len(positions)):
            positions[i, :] = np.dot(positions[i,:], rotation_matrix)
    return basex, basey, basez, positions

def _positive_vectors(basex, basey, basez, positions):
    if basex[0]<0:
        basex = basex*-1
        positions[:, 0] = positions[:, 0]*-1
    if basey[1]<0:
        basey = basey*-1
        positions[:, 1] = positions[:, 1]*-1
    if basez[2]<0:
        basez = basez*-1
        positions[:, 2] = positions[:, 2]*-1
    for i in range(3):
        if np.abs(basex[i])<.00001:
            basex[i] = 0
        if np.abs(basey[i])<.00001:
            basey[i] = 0
        if np.abs(basez[i])<.00001:
            basez[i] = 0
    return basex, basey, basez, positions

def _axis_swap(swap, basex, basey, basez, positions):
    if swap=='xy' or swap=='yx':
        temp = basex
        basex = basey
        basey = temp
        temp = positions[:, 0 ]
        positions[:,0] = positions[:,1]
        positions[:,1] = temp
    if swap=='xz' or swap=='zx':
        temp = basex
        basex = basez
        basez = temp
        temp = positions[:, 0 ]
        positions[:,0] = positions[:,2]
        positions[:,2] = temp
    if swap=='zy' or swap=='yz':
        temp = basez
        basez = basey
        basey = temp
        temp = positions[:, 2]
        positions[:,2] = positions[:,1]
        positions[:,1] = temp
    return basex, basey, basez, positions

def geometry_to(filein):
        file = open(filein)
        lines = file.readlines()
        file.close()

        basis = [l for l in lines if l[:14]=='lattice_vector']        
        core = [l for l in lines if l[:4]=='atom']

        name = 'name\n'
        basis = [l[15:] for l in basis]
        basex = np.array([float(x) for x in basis[0].split()])
        basey = np.array([float(x) for x in basis[1].split()])
        basez = np.array([float(x) for x in basis[2].split()])
        
        labels = [n.split(" ")[-1][:-1] for n in core]
        labels = np.array(labels)
        atoms_type = np.unique(labels)
        atoms_quantity = []
        for atom in atoms_type:
            atoms_quantity.append(np.sum(labels==atom))
        atoms_variety = len(atoms_quantity)
        
        positions_splited = []
        for at in atoms_type:
            position_one_atom_type = []
            for c in core:
                data = c.split()[1:]
                if data[-1]==at:
                    position_one_atom_type.append(data[:3])
            positions_splited.append(np.array(position_one_atom_type, float))
        return name, basex, basey, basez, atoms_type, atoms_quantity, atoms_variety, positions_splited

def lammps_to(filein):
    file = open(filein)
    lines = file.readlines()
    file.close()
    
    name = lines[0]
    atoms_quantity_total = int(lines[2].split()[0])
    atoms_variety = int(lines[3].split()[0])
    atoms_type = lines[12:12+atoms_variety]
    atoms_type = [float(a.split()[1]) for a in atoms_type]
    atoms_type = [mases_to_symbols[a] for a in atoms_type]
    
    basis = lines[5:9]
    basis = [b.split() for b in basis]

    xl = np.asanyarray(basis[0][:2],float)
    yl = np.asanyarray(basis[1][:2],float)
    zl = np.asanyarray(basis[2][:2],float)
    xy = float(basis[3][0])
    xz = float(basis[3][1])
    yz = float(basis[3][2])
    
    basex = np.array([xl[1]-xl[0], 0, 0])
    basey = np.array([xy, yl[1]-yl[0], 0])
    basez = np.array([xz, yz, zl[1]-zl[0]])
    
    positions = lines[15+atoms_variety:15+atoms_variety+atoms_quantity_total]
    positions = np.asanyarray([p.split()[:5] for p in positions])
    pos_idx = positions[:,0].astype(int)-1
    pos_type = positions[:,1].astype(int)-1
    positions = positions[:,2:].astype(float)
    pos_type = pos_type[pos_idx]
    positions = positions[pos_idx,:]
    
    positions_splited = []
    atoms_quantity = []
    for a in range(atoms_variety):
        positions_splited.append(positions[pos_type==a,:])
        atoms_quantity.append(np.sum(pos_type==a))
    return name, basex, basey, basez, atoms_type, atoms_quantity, atoms_variety, positions_splited

def to_poscar(
        name,
        basex, basey, basez,
        atoms_type, atoms_quantity, atoms_variety,
        positions_splited,
        fileout,
        ):    
    poscar = [name]
    poscar.append('1.0\n')
    poscar.append('\t'+str(basex[0])+'\t'+str(basex[1]) +'\t' + str(basex[2]) + '\n')
    poscar.append('\t'+str(basey[0])+'\t'+str(basey[1]) +'\t' + str(basey[2]) + '\n')
    poscar.append('\t'+str(basez[0])+'\t'+str(basez[1]) +'\t' + str(basez[2]) + '\n')
    poscar.append('\t'+'\t'.join(atoms_type) + '\n')
    poscar.append('\t'+'\t'.join([str(a) for a in atoms_quantity]) + '\n')
    poscar.append('Cartesian\n')
    for positions in positions_splited:
        for p in positions:
            poscar.append(str(p[0]) + '\t' + str(p[1]) + '\t' + str(p[2]) + '\t' + '\n')
    poscar = "".join(poscar)
    
    file = open(fileout, 'w')
    file.write(poscar)
    file.close()

def poscar_to(file_in):
    file = open(file_in)
    lines = file.readlines()
    file.close()

    name, basex, basey, basez, atoms_tot, positions = _parsing_poscar(lines)
    basex, basey, basez, positions = _rotating_positions(basex, basey, basez, positions)
    basex, basey, basez, positions = _positive_vectors(basex, basey, basez, positions)
    
    atoms_type = list(filter(None,lines[5][:-1].split()))
    atoms_quantity = np.array(list(filter(None,lines[6][:-1].split())), int)
    atoms_variety = len(atoms_quantity)
    positions = lines[8:]
    positions = np.asanyarray([p.split() for p in positions], float)
    
    positions_splited = []
    nl = 0
    for i in range(atoms_variety):
        positions_splited.append(positions[nl:nl+atoms_quantity[i]])
        nl = nl + atoms_quantity[i]
    
    return name, basex, basey, basez, atoms_type, atoms_quantity, atoms_variety, positions_splited

def desc_reader(fpath, norm_folder):
    #creating empty matrix
    data = poscar_to(fpath+'/POSCAR')
    atom_quantity = data[5]

    data = np.load(fpath+'/soap.npz')
    norm = np.load(norm_folder+'/norm.npz')
    desc_mean = norm['desc_mean']
    desc_std = norm['desc_std']
    desc = (data['desc']-desc_mean)/desc_std

    shape = desc.shape
    desc = np.reshape(desc, shape[0]*shape[1])
    return desc, shape, atom_quantity

def gkm(desc, sigma, l, atom_quant, K):
    k = np.zeros((K,K))
    for j in range(K):
        cd = (desc - desc[j])**2
        k[j,:] = sigma*np.exp(-np.sum(cd/(2*l*l), 1))
    return k

def gk(desc1, desc2, sigma, l, K, Pns):
    ks = np.zeros((K, Pns))
    for i in range(K):
        for j in range(Pns):
            cd = (desc1[i,:] - desc2[j,:])**2
            ks[i,j] = sigma*np.exp(-np.sum(cd/(2*l*l)))
    return ks

def _calc_alfa_pieces(fpath_training, training_set, sigma, l, se):
    training_set = np.sort(training_set)
    N = len(training_set)
    desc = []
    atom_quant = []
    Pn = []
    
    file = open(fpath_training+'fe.dat')
    data = file.readlines()[1:]
    file.close()
    
    fe_samples = [d.split()[0] for d in data]
    idx = np.argsort(fe_samples)
    F = np.array([d.split()[1] for d in data], dtype=float)[idx]

    for ts in training_set:
        Pn.append(np.sum(poscar_to(fpath_training+ts+'/POSCAR')[5]))
        _desc, shape, _atom_quant = desc_reader(fpath_training+ts, fpath_training)
        _desc = np.split(_desc, Pn[-1])
        desc = desc + _desc
        atom_quant = atom_quant + _atom_quant.tolist()
    desc = np.asanyarray(desc)
    K = np.sum(Pn)

    #calculating L matrix
    L = np.zeros((K, N))
    step = 0
    for n in range(N):
        L[step:step+Pn[n],n] = 1
        step = step + Pn[n]

#    calculating covariance
    k = gkm(desc, sigma, l, atom_quant, K)
    len_k = len(k)

    #calculating alpha
    alfa_braket = np.matmul(L.T, np.matmul(k, L))
    alfa_braket = alfa_braket + (se**2)*np.identity(len(alfa_braket))
    try:
        alfa_cholesky = np.linalg.cholesky(alfa_braket)
        alfa_cholesky_H = alfa_cholesky.T.conj()
        alfa_braket_inv = np.matmul(
            np.linalg.inv(alfa_cholesky_H),
            np.linalg.inv(alfa_cholesky)
            )
    except:
        print('cholesky decomposition failed')
        alfa_braket_inv = np.linalg.inv(alfa_braket)
    return alfa_braket, alfa_braket_inv, F, len_k, L, K, desc
    
def calc_mlh(fpath_training, training_set, sigma, l, se):
    alfa_braket, alfa_braket_inv, F, len_k, L, K, desc = _calc_alfa_pieces(
            fpath_training, training_set, 
            sigma, l, se,
            )
#    #minimise that!
    mlh =   (
            .5*np.log(np.abs(np.linalg.det(alfa_braket))) + 
            .5*np.matmul(F.T, np.matmul(alfa_braket_inv, F)) + 
            .5*len_k*np.log(2*np.pi)
            )
    return mlh

def calc_alfa(fpath_training, training_set, sigma, l, se):
    alfa_braket, alfa_braket_inv, F, len_k, L, K, desc = _calc_alfa_pieces(
            fpath_training, training_set, 
            sigma, l, se,
            )
    alfa = np.matmul(
            L,
            np.matmul(alfa_braket_inv, F),
            )
    return alfa, desc, K

def calc_prediction(fpath_training, fpath_predication, prediction_set, alfa, desc, K, sigma, l):
    Fp = []
    for ps in prediction_set:
        #reading data
        Pns = np.sum(poscar_to(fpath_predication + ps+'/POSCAR')[5])
        _desc, shape, _atom_quant = desc_reader(fpath_predication + ps, fpath_training)
        descs = np.asanyarray(np.split(_desc, Pns))
        
        #calculating covariance
        ks = gk(desc, descs, sigma, l, K, Pns)

#        #calculatig free energy - prediction
        Fs_pr = []
        for p in range(Pns):
            Fs_pr.append(np.sum(ks[:,p]*alfa[:]))
        Fs_pr = np.sum(Fs_pr)
        Fp.append(Fs_pr)

    return np.array(Fp)
