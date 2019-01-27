#!/usr/bin/python -u

"""
Code de diagonalisation exacte dans la base a np particules du tight-binding
sur une chaine:

	H = - Kina * ham_kin + Pot * ham_pot

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

    bond = numpy.zeros((ns, 5), dtype="int")
    for s in numpy.arange(ns):

        bond[s][0] = s #itself
        bond[s][1] = right(s, nx, ny)
        bond[s][2] = left(s, nx, ny)
        bond[s][3] = up(s, nx, ny)
        bond[s][4] = down(s, nx, ny)

    return bond


# For 2D square lattice without spins
def bond_2D_triangular(nx, ny):

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
    for x in numpy.arange(2 ** ns):
        size += (bin(x).count("1") == np) # Count those with np occupied sites

    print("Hilbert size : {0}".format(size))
    basis = numpy.zeros(size, dtype="int")

    size = 0

    # Generate the entire Hilbert space for a system with ns sites.
    for x in numpy.arange(2 ** ns):
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


def create_cluster(nx, ny, np, particle='boson', lattice='1D'):

    ns = nx*ny
    basis, size = generate_hilbert_space(ns, np)

    ham_pot = numpy.zeros((size, size), dtype='double')
    ham_kin = numpy.zeros((size, size), dtype='double')

    if lattice is '1D':
        bond = generate_bonds_1D(ns)
    elif lattice is 'square':
        bond = bond_2D_square(nx, ny)
    elif lattice is 'triangular':
        bond = bond_2D_triangular(nx, ny)

# For more comments c.f. notes in Squid notebook: Pro Num/Rascunho pag 3, 4, 5
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

                    sign = commutation(state, s0, s1, particle=particle)

                    ham_kin[w, numpy.where(basis == state ^ w0 ^ w1)] += sign

            ham_pot[w, w] = Diag

    return basis, ham_pot, ham_kin

nx = 5		# linear size
ny = 6
np = 4
Kina = 1.00			# hopping term
Pot = 0	# n.n. interaction
particle = 'fermion'

ns = nx * ny

basis, ham_pot, ham_kin = create_cluster(nx, ny, np,
                                                particle, lattice='triangular')

start = time.clock()

#print "# ======== Diagonalization : brute force "
#EigenEnergies , EigenVectors = eigh( Pot * ham_pot -  Kina * ham_kin )
#stop  = time.clock()
#print EigenEnergies[ 0 ], " in ",stop-start," seconds"

start = time.clock()
print("# ======== Diagonalization : Lanczos")
EigenEnergies, EigenVectors = eigs(Pot * ham_pot - Kina * ham_kin,
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


# Correlation matrix
corre_mat = numpy.ones((ns, ns)) * numpy.nan
for s in numpy.arange(ns):
    for w in numpy.arange(ns):
        corre_mat[s, w] = density_vect[s] * density_vect[w]

print(corre_mat)
print(numpy.mean(corre_mat, axis=1))
