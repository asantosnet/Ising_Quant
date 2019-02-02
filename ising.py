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


def calculate_bonds_field(field, row, column, data,
                          element_id, basis, bond):

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

    element_id = element_id + n_non_diagonal

    return row, column, data, element_id


def calculate_bonds_f_neig(echange, row, column,
                           data, element_id, basis, bond):

    for w in numpy.arange(len(basis)):
        state = basis[w]
        Diag = 0

        for b in range(len(bond)):
            s0 = bond[b, 0]
            w0 = 1 << s0
            for i in numpy.arange(1, bond.shape[1]):
                s1 = bond[b, i]
                w1 = 1 << s1

                if (state & w0 > 0) == (state & w1 > 0):
                    Diag += +1
                else:
                    Diag += - 1

        row[element_id] = w
        column[element_id] = w
        data[element_id] = Diag * echange
        element_id += 1

    return row, column, data, element_id


def calculate_bonds_f_neig_opt(echange, row, column,
                               data, element_id, basis, bond):

    n_elements = len(basis)

    eles = numpy.arange(element_id,
                        element_id + n_elements)

    row[eles] = basis
    column[eles] = basis

    # True if spin is up or False if down
    spin_site = (numpy.bitwise_and(
            numpy.repeat(basis, bond.size),
            numpy.tile(numpy.repeat(
                    numpy.left_shift(1, bond[:, 0]),
                    bond.shape[1]),
                       len(basis))) > 0)

    # True if spin is up or False if down
    spin_neig = (numpy.bitwise_and(
            numpy.repeat(basis, bond.size),
            numpy.tile(numpy.left_shift(
                    1, numpy.concatenate(bond)),
                        len(basis))) > 0)

    # Find all combinations (up_up, d_d_, u_d, d_u)
    up_up = spin_site * spin_neig
    down_down = (numpy.logical_not(spin_site)
                 * numpy.logical_not(spin_neig))

    up_down = (spin_site * numpy.logical_not(spin_neig))
    down_up = (numpy.logical_not(spin_site) * spin_neig)

    Diag = (up_up.astype(int) + down_down.astype(int)
            - up_down.astype(int) - down_up.astype(int))

    Diag = numpy.reshape(Diag, (n_elements, bond.size))

    # Reduce of bond.shape[0] because in the spin_neig
    # the site itself is also there. This adds a factor
    # of bond.shape[0] to the sum. (i.e the number
    # of sites in the system)
    Diag = numpy.sum(Diag, axis=1) - bond.shape[0]
    data[eles] = Diag * echange

    return row, column, data, element_id


def gen_hamiltonian_Ising(size, bond, basis, factors,
                          nx, ny):

    n_el = size + (size * nx * ny)

    row = numpy.zeros(n_el, dtype=int)
    column = numpy.zeros(n_el, dtype=int)
    data = numpy.zeros(n_el, dtype='double')

    echange, field = factors
    element_id = 0

    print('Calculating bonds - field')

    row, column, data, element_id  = calculate_bonds_field(
            field, row, column, data, element_id,
            basis, bond)

    print('Calculating bonds - Fneig interaction')

    (row, column,
     data, element_id) = calculate_bonds_f_neig_opt(
            echange, row, column, data, element_id,
            basis, bond)

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
nx = 4		# linear size
ny = 4
echange = 1
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
