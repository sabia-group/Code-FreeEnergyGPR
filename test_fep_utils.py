import os

import numpy as np
import pytest

import fep_utils as fu

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.join(HERE, 'training') + '/'
PREDICTION = os.path.join(HERE, 'prediction') + '/'
# fe.dat lists all training samples, so the full set must be used
SAMPLES = sorted(fu.list_of_directories(TRAINING))


def test_list_of_files(tmp_path):
    (tmp_path / 'a.txt').write_text('x')
    (tmp_path / 'sub').mkdir()
    assert fu.list_of_files(str(tmp_path)) == ['a.txt']


def test_list_of_directories(tmp_path):
    (tmp_path / 'a.txt').write_text('x')
    (tmp_path / 'sub').mkdir()
    assert fu.list_of_directories(str(tmp_path)) == ['sub']


def test_angle():
    assert fu.angle([1, 0, 0], [0, 1, 0]) == pytest.approx(np.pi / 2)
    assert fu.angle([1, 0, 0], [2, 0, 0]) == pytest.approx(0)
    assert fu.angle([1, 0, 0], [-1, 0, 0]) == pytest.approx(np.pi)


def test_u_rotate_runs():
    # u_rotate currently builds the matrix but does not return it
    assert fu.u_rotate(0.3, np.array([0., 0., 2.])) is None


@pytest.mark.xfail(reason='u_rotate does not return the rotation matrix')
def test_u_rotate_matrix():
    rot = fu.u_rotate(np.pi / 2, np.array([0., 0., 1.]))
    assert np.allclose(rot, [[0, -1, 0], [1, 0, 0], [0, 0, 1]])


POSCAR_LINES = [
    'name\n', '2.0\n',
    '1 0 0\n', '0 1 0\n', '0 0 1\n',
    'C H\n', '1 2\n', 'Cartesian\n',
    '0.0 0.0 0.0 \n', '1.0 1.0 1.0 \n', '2.0 2.0 2.0 \n',
]


def test_parsing_poscar():
    name, bx, by, bz, tot, pos = fu._parsing_poscar(POSCAR_LINES)
    assert name == 'name\n'
    assert np.allclose(bx, [2, 0, 0])
    assert np.allclose(by, [0, 2, 0])
    assert np.allclose(bz, [0, 0, 2])
    assert tot == 3
    assert pos.shape == (3, 3)


def test_rotating_positions_noop_for_aligned_cell():
    bx, by, bz = np.eye(3) * 2
    pos = np.arange(9, dtype=float).reshape(3, 3)
    rbx, rby, rbz, rpos = fu._rotating_positions(bx, by, bz, pos.copy())
    assert np.allclose(rbx, bx) and np.allclose(rby, by) and np.allclose(rbz, bz)
    assert np.allclose(rpos, pos)


def test_positive_vectors():
    bx = np.array([-1., 0., 0.])
    by = np.array([0., -2., 0.])
    bz = np.array([0.000001, 0., -3.])
    pos = np.ones((2, 3))
    bx, by, bz, pos = fu._positive_vectors(bx, by, bz, pos)
    assert bx[0] == 1 and by[1] == 2 and bz[2] == 3
    assert bz[0] == 0  # tiny components are zeroed
    assert np.allclose(pos, -1)


def test_axis_swap_bases():
    bx, by, bz = np.eye(3)
    pos = np.array([[1., 2., 3.]])
    sbx, sby, sbz, _ = fu._axis_swap('xy', bx, by, bz, pos.copy())
    assert np.allclose(sbx, by) and np.allclose(sby, bx)
    sbx, sby, sbz, _ = fu._axis_swap('xz', bx, by, bz, pos.copy())
    assert np.allclose(sbx, bz) and np.allclose(sbz, bx)
    sbx, sby, sbz, spos = fu._axis_swap('none', bx, by, bz, pos.copy())
    assert np.allclose(spos, pos)


@pytest.mark.xfail(reason='positions swap uses a view as temp, so one column is duplicated')
def test_axis_swap_positions():
    bx, by, bz = np.eye(3)
    pos = np.array([[1., 2., 3.]])
    _, _, _, spos = fu._axis_swap('xy', bx, by, bz, pos.copy())
    assert np.allclose(spos, [[2, 1, 3]])


def test_geometry_to(tmp_path):
    f = tmp_path / 'geometry.in'
    f.write_text(
        'lattice_vector 1.0 0.0 0.0\n'
        'lattice_vector 0.0 2.0 0.0\n'
        'lattice_vector 0.0 0.0 3.0\n'
        'atom 0.0 0.0 0.0 H\n'
        'atom 1.0 1.0 1.0 C\n'
        'atom 2.0 2.0 2.0 H\n'
    )
    name, bx, by, bz, types, qty, variety, pos = fu.geometry_to(str(f))
    assert list(types) == ['C', 'H']
    assert qty == [1, 2]
    assert variety == 2
    assert np.allclose(by, [0, 2, 0])
    assert np.allclose(pos[1], [[0, 0, 0], [2, 2, 2]])


def test_lammps_to(tmp_path):
    f = tmp_path / 'data.lammps'
    f.write_text(
        'name\n'
        '\n'
        '3 atoms\n'
        '2 atom types\n'
        '\n'
        '0.0 4.0 xlo xhi\n'
        '0.0 5.0 ylo yhi\n'
        '0.0 6.0 zlo zhi\n'
        '0.5 0.6 0.7 xy xz yz\n'
        '\n'
        '\n'
        'Masses\n'
        '1 12.0107\n'
        '2 1.0080\n'
        '\n'
        'Atoms\n'
        '\n'
        '1 1 0.0 0.0 0.0\n'
        '2 2 1.0 1.0 1.0\n'
        '3 2 2.0 2.0 2.0\n'
    )
    name, bx, by, bz, types, qty, variety, pos = fu.lammps_to(str(f))
    assert types == ['C', 'H']
    assert variety == 2
    assert list(qty) == [1, 2]
    assert np.allclose(bx, [4, 0, 0])
    assert np.allclose(by, [0.5, 5, 0])
    assert np.allclose(bz, [0.6, 0.7, 6])
    assert pos[1].shape == (2, 3)


def test_to_poscar_roundtrip(tmp_path):
    out = tmp_path / 'POSCAR'
    pos = [np.array([[0., 0., 0.]]), np.array([[1., 1., 1.], [2., 2., 2.]])]
    fu.to_poscar(
        'name\n',
        np.array([1., 0., 0.]), np.array([0., 2., 0.]), np.array([0., 0., 3.]),
        ['C', 'H'], [1, 2], 2, pos, str(out),
    )
    lines = out.read_text().splitlines()
    assert lines[0] == 'name'
    assert lines[1] == '1.0'
    assert lines[5].split() == ['C', 'H']
    assert lines[6].split() == ['1', '2']
    assert len(lines) == 8 + 3

    name, bx, by, bz, types, qty, variety, ps = fu.poscar_to(str(out))
    assert types == ['C', 'H']
    assert list(qty) == [1, 2]
    assert variety == 2
    assert np.allclose(bz, [0, 0, 3])
    assert np.allclose(ps[1], pos[1])


def test_poscar_to_real_file():
    name, bx, by, bz, types, qty, variety, ps = fu.poscar_to(
        TRAINING + 'ANTCEN10_r_0/POSCAR')
    assert variety == len(types) == len(ps)
    assert [len(p) for p in ps] == list(qty)


def test_desc_reader():
    desc, shape, qty = fu.desc_reader(TRAINING + SAMPLES[0], TRAINING)
    assert desc.shape == (shape[0] * shape[1],)
    assert sum(qty) == shape[0]


def test_gkm():
    desc = np.array([[0., 0.], [1., 0.], [0., 2.]])
    k = fu.gkm(desc, 2.0, 1.0, None, 3)
    assert k.shape == (3, 3)
    assert np.allclose(np.diag(k), 2.0)
    assert np.allclose(k, k.T)
    assert k[0, 1] == pytest.approx(2 * np.exp(-0.5))


def test_gk():
    d1 = np.array([[0., 0.], [1., 0.]])
    d2 = np.array([[0., 0.], [0., 2.], [1., 0.]])
    ks = fu.gk(d1, d2, 2.0, 1.0, 2, 3)
    assert ks.shape == (2, 3)
    assert ks[0, 0] == pytest.approx(2.0)
    assert ks[0, 1] == pytest.approx(2 * np.exp(-2))
    assert ks[1, 2] == pytest.approx(2.0)


def test_calc_alfa_pieces():
    braket, inv, F, len_k, L, K, desc = fu._calc_alfa_pieces(
        TRAINING, SAMPLES, 1.0, 5.0, 0.1)
    assert braket.shape == (len(SAMPLES),) * 2
    assert np.allclose(braket @ inv, np.identity(len(SAMPLES)), atol=1e-6)
    assert F.shape == (len(SAMPLES),)
    assert L.shape == (K, len(SAMPLES))
    assert len_k == K == len(desc)


def test_calc_mlh():
    mlh = fu.calc_mlh(TRAINING, SAMPLES, 1.0, 5.0, 0.1)
    assert np.isfinite(mlh)


def test_calc_alfa():
    alfa, desc, K = fu.calc_alfa(TRAINING, SAMPLES, 1.0, 5.0, 0.1)
    assert alfa.shape == (K,)
    assert len(desc) == K


def test_calc_prediction():
    alfa, desc, K = fu.calc_alfa(TRAINING, SAMPLES, 1.0, 5.0, 0.1)
    fp = fu.calc_prediction(
        TRAINING, PREDICTION, ['BENZENE13'], alfa, desc, K, 1.0, 5.0)
    assert fp.shape == (1,)
    assert np.isfinite(fp).all()
