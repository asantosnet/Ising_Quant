#!/usr/bin/python -u

"""
Code de diagonalisation exacte dans la base spin up down completement rempli

	* eigh : brute force
	* eigs : Lanczos
	* calcul des densites moyennes sur site pour montrer aux etudiants
	  l'invariance par translation des solutions et les effets des
	  degenerescences
	* faire implementer le signe fermionique (present dans ce code mais a
	  enlever quand distribuer)
	* faire implementer les correlations charge-chare

"""
import time
import numpy
from scipy.linalg import eigh
from scipy.sparse.linalg import eigs
from matplotlib import pyplot as plt
import scipy.sparse as sps

# Define operations for a square lattice :
def up(x, nx, ny):
    '''
        Find up neighbourd square lattice
        Input:
            x :  lattice point
            nx and ny: dimensions of the lattice
        Returns:
            y :  lattice point
    '''
    return (x + nx) % (nx * ny)


def down(x, nx, ny):
    '''
        Find down neighbourd square lattice
        Input:
            x :  lattice point
            nx and ny: dimensions of the lattice
        Returns:
            y :  lattice point
    '''
    return (x - nx) % (nx * ny)


def left(x, nx, ny):
    '''
        Find left neighbourd square lattice
        Input:
            x :  lattice point
            nx and ny: dimensions of the lattice
        Returns:
            y :  lattice point
    '''
    return nx * (x // nx) + (x - 1) % nx


def right(x, nx, ny):
    '''
        Find right neighbourd square lattice
        Input:
            x :  lattice point
            nx and ny: dimensions of the lattice
        Returns:
            y :  lattice point
    '''
    return nx * (x // nx) + (x + 1) % nx


# Creat the bond for 1D lattice
def generate_bonds_1D(ns):

    bond = numpy.zeros((ns, 2), dtype="int")
    for s in numpy.arange(ns):
        bond[s][0] = s
        bond[s][1] = (s + 1) % ns

    return bond


# For 2D square lattice without spins
def bond_2D_square(nx, ny):

    bonds_functions = [right, left, up, down]

    ns = nx * ny
    bond = numpy.zeros((ns, 1 + len(bonds_functions)), dtype="int")

    for s in numpy.arange(ns):

        # itself
        bond[s][0] = s
        for bond_num, bond_fun in enumerate(bonds_functions):
            # first neig
            bond[s][bond_num + 1] = bond_fun(s, nx, ny)

    return bond

# For 2D square lattice without spins
def bond_2D_triangular(nx, ny):

    ns = nx * ny
    bond = numpy.zeros((ns, 7), dtype="int")
    for s in numpy.arange(ns):

        bond[s][0] = s #itself
        bond[s][1] = right(s, nx, ny)
        bond[s][2] = left(s, nx, ny)
        bond[s][3] = up(s, nx, ny)
        bond[s][4] = down(s, nx, ny)

        bond[s][5] = down(right(s, nx, ny), nx, ny)
        bond[s][6] = up(left(s, nx, ny), nx, ny)

    return bond


# Create the Hilbert space basis
def generate_hilbert_space(ns):

    size = 2 ** ns

    print("Hilbert size : {0}".format(size))
    basis = numpy.arange(size)

    return basis, size

# TODO: Comment prendre en consideration dans l'axe x
# TODO: solve memory error ? Use sparse matrix
def gen_hamiltonian_Ising(size, bond, basis, factors,
                          nx, ny):

    n_el = size + (size * nx * ny)

    row = numpy.zeros(n_el , dtype = int)
    column = numpy.zeros(n_el , dtype = int)
    data = numpy.zeros(n_el , dtype = 'double')

    echange, champ = factors
    element_id = 0

    print('Calculating bonds')

    t1 = time.clock()

    n_non_diagonal = len(bond) * len(basis)
    eles = numpy.arange(element_id,
                        element_id + n_non_diagonal)

    row[eles] = numpy.repeat(numpy.arange(len(basis)),
                             len(bond))

    column[eles] = numpy.bitwise_xor(
            numpy.repeat(basis, len(bond)),
            numpy.tile(
                    numpy.left_shift(1, bond[:, 0]),
                    len(basis)))
    data[eles] = champ

    print(column)
    t2 = time.clock()

    print('{}'.format(t2 - t1))

    row = numpy.zeros(n_el , dtype = int)
    column = numpy.zeros(n_el , dtype = int)
    data = numpy.zeros(n_el , dtype = 'double')


    t1 = time.clock()

    for w in numpy.arange(len(basis)):
        state = basis[w]
#        Diag = len(bond) - 1

        for b in range(len(bond)):
            s0 = bond[b, 0]
            w0 = 1 << s0
#            for i in numpy.arange(1, bond.shape[1]):
#                s1 = bond[b, i]
#                w1 = 1 << s1
#
#                if (state & w0 > 0) != (state & w1 > 0):
#                    Diag += - 2

            # Add the field operator
            row[element_id] = w
            column[element_id] = state ^ w0
            data[element_id] = champ
            element_id += 1

#        eles = numpy.arange(element_id,
#                            element_id + len(bond))
#        row[eles] = numpy.ones(len(bond)) * w
#        column[eles] = numpy.bitwise_xor(state,
#              numpy.left_shift(1, bond[:, 0]))
#        data[eles] = champ

#        row[element_id] = w
#        column[element_id] = w
#        data[element_id] = Diag
#
#        element_id += 1

    t2 = time.clock()
    print('{}'.format(t2 - t1))
    print(column)

    raise


    print('Making sparse')

    ham_ech = sps.csc_matrix((data * echange,
                                (row, column)),
                               shape = (size, size))

    return ham_ech


def create_cluster(nx, ny, factors, lattice='1D'):

    ns = (nx * ny)
    basis, size = generate_hilbert_space(ns)

    if lattice is '1D':
        bond = generate_bonds_1D(ns)
    elif lattice is 'square':
        bond = bond_2D_square(nx, ny)
    elif lattice is 'triangular':
        bond = bond_2D_triangular(nx, ny)

    hamiltonian = gen_hamiltonian_Ising(
            size, bond, basis, factors, nx, ny)

    return basis, hamiltonian

#nx = 5		# linear size
#ny = 5
#np = 12
nx = 2		# linear size
ny = 2
echange = -1
champ = 0
factors = [echange, champ]

ns = (nx * ny)

basis, hamiltonian = create_cluster(
        nx, ny, factors=factors,
        lattice='square')


start = time.clock()
print("# ======== Diagonalization : Lanczos")
EigenEnergies, EigenVectors = eigs(hamiltonian,
                                   1, None, None, 'SR', None, None, None,
                                   1.e-5)
stop = time.clock()
print("{0} in {1} seconds".format(EigenEnergies[0], stop-start))

print(EigenEnergies)

density_vect = numpy.ones(ns)
for s in numpy.arange(ns):
    dens = 0.0
    for w in numpy.arange(len(EigenVectors[:, 0])):
        if (((basis[w] & 1 << 0) != 0) and ((basis[w] & 1 << s) != 0)):
#        if (((basis[w] & 1 << s) != 0)):
            dens += EigenVectors[w, 0] * numpy.conj(EigenVectors[w, 0])

    density_vect[s] = dens
    print("{0} \t {1}".format(s, dens))

## Correlation matrix
#corre_mat = numpy.ones((ns, ns)) * numpy.nan
#for s in numpy.arange(ns):
#    for w in numpy.arange(ns):
#        corre_mat[s, w] = density_vect[s] * density_vect[w]
#
#print(corre_mat)
#print(numpy.mean(corre_mat, axis=1))
