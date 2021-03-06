#!/usr/bin/python -u

"""
Code de diagonalisation exacte dans la base a np particules

	* eigh : brute force
	* eigs : Lanczos
	* calcul des densites moyennes sur site pour montrer aux etudiants
	  l'invariance par translation des solutions et les effets des
	  degenerescences
	* faire implementer le signe fermionique (present dans ce code mais a
	  enlever quand distribuer)
	* faire implementer les correlations charge-chare

"""
import math, time
import cmath
import numpy
import sys
from scipy.linalg import eigh
from scipy.sparse.linalg import eigs

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

# For 2D square lattice with spins
def bond_2D_square_spin(nx, ny):

    bonds_functions = [right, left, up, down,
                       lambda s, nx, ny: right(s, nx, ny) + (nx * ny),
                       lambda s, nx, ny: left(s, nx, ny) + (nx * ny),
                       lambda s, nx, ny: up(s, nx, ny) + (nx * ny),
                       lambda s, nx, ny: down(s, nx, ny) + (nx * ny)]

    bond = numpy.zeros((2 * (nx * ny),
                        1 + len(bonds_functions)), dtype="int")

    for s in numpy.arange(nx * ny):

        # itself
        bond[s][0] = s
        bond[s + (nx * ny)][0] = s + (nx * ny)

        for bond_num, bond_fun in enumerate(bonds_functions):
            # spin up-up et up-down
            bond[s][bond_num + 1] = bond_fun(s, nx, ny)
            # spin down-up et down-down
            bond[s + (nx * ny)][bond_num + 1] = bond_fun(s, nx, ny)

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
def generate_hilbert_space(ns, np):

    size = 0
    # Generate the entire Hilbert space for a system with ns sites.
    #TODO Remove this numpy arange and create each eigenstate iteratively ?
    for x in range(2 ** ns):
        size += (bin(x).count("1") == np) # Count those with np occupied sites

    print("Hilbert size : {0}".format(size))
    basis = numpy.zeros(size, dtype="int")

    size = 0

    # Generate the entire Hilbert space for a system with ns sites.
    for x in range(2 ** ns):
        if bin(x).count("1") == np: # Select those with np occupied sites
            basis[size] = x # Add it to the the basis vector
            size += 1

    return basis, size


# Apply (anti)-commutation relations
def commutation(state, s0, s1, particle=None):

    sign1 = + 1

    if particle is 'boson':
        sign1 = + 1

    elif particle is 'fermion':
        if s0 < s1:
            for c in numpy.arange(s0 + 1, s1):
                if (state & (1 << c) > 0):
                    sign1=sign1 * (-1)
        if s1 < s0:
            for c in numpy.arange(s1 + 1, s0):
                if (state & (1 << c) > 0):
                    sign1 = sign1 * (-1)

    return sign1


# H =  kina * ham_kin + Pot * ham_pot
def gen_hamiltonian_serie4(size, bond, basis, factors,
                           particle):

    ham_pot = numpy.zeros((size, size), dtype='double')
    ham_kin = numpy.zeros((size, size), dtype='double')

    kina, pot = factors

    for w in numpy.arange(len(basis)):
        state = basis[w]
        Diag = 0
        for b in range(len(bond)):
            s0 = bond[b, 0]
            w0 = 1 << s0
            for i in numpy.arange(1, bond.shape[1]):
                s1 = bond[b, i]
                w1 = 1 << s1
                Diag += (state & w0 > 0) * (state & w1 > 0)
                if (state & w0 > 0) != (state & w1 > 0):

                    sign = commutation(state, s0, s1,
                                       particle=particle)

                    ham_kin[w, numpy.where(basis == state ^ w0 ^ w1)] += sign

            ham_pot[w, w] = Diag

    return (pot * ham_pot + kina * ham_kin)


# TODO: Comment prendre en consideration dans l'axe x
# TODO: solve memory error ?
def gen_hamiltonian_Ising(size, bond, basis, factors,
                          nx, ny):

    ham_ech = numpy.zeros((size, size), dtype='double')
    #ham_kin = numpy.zeros((size, size), dtype='double')

    echange, champ = factors

    for w in numpy.arange(len(basis)):
        state = basis[w]
        Diag = 0
        for b in range(len(bond)):
            s0 = bond[b, 0]
            w0 = 1 << s0
            for i in numpy.arange(1, bond.shape[1]):
                s1 = bond[b, i]
                w1 = 1 << s1
                sign = (-1) ** ((s0 // (nx * ny))
                                 + (s1 // (nx * ny)))
                Diag += ((state & w0 > 0)
                         * (state & w1 > 0)) * sign

            ham_ech[w, w] = Diag

    return (echange * ham_ech)


def create_cluster(nx, ny, np, factors, particle='boson',
                   lattice='1D'):

    ns = 2 * (nx * ny)
    basis, size = generate_hilbert_space(ns, np)

    if lattice is '1D':
        bond = generate_bonds_1D(ns)
    elif lattice is 'square':
        bond = bond_2D_square(nx, ny)
    elif lattice is 'triangular':
        bond = bond_2D_triangular(nx, ny)
    elif lattice is 'square_spin':
        bond = bond_2D_square_spin(nx, ny)

#    hamiltonian = gen_hamiltonian_serie4(size, bond, basis,
#                                         factors, particle)

    hamiltonian = gen_hamiltonian_Ising(size, bond,
                                        basis, factors,
                                        nx, ny)
    return basis, hamiltonian

#nx = 5		# linear size
#ny = 5
#np = 12
nx = 2		# linear size
ny = 2
np = 4
kina = -1.00			# hopping term
pot = 0	# n.n. interaction
factors = [kina, pot]
echange = 1
champ = 0
factors = [echange, champ]
particle = 'fermion'

ns = 2 * (nx * ny)

basis, hamiltonian = create_cluster(
        nx, ny, np, factors=factors,
        particle=particle, lattice='square_spin')

start = time.clock()

#print "# ======== Diagonalization : brute force "
#EigenEnergies , EigenVectors = eigh( pot * ham_pot -  kina * ham_kin )
#stop  = time.clock()
#print EigenEnergies[ 0 ], " in ",stop-start," seconds"

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
