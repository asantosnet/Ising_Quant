#!/usr/bin/python -u

"""
Code de diagonalisation exacte dans la base spin up down completement rempli

	* eigh : brute force
	* eigs : Lanczos
	* calcul des densites moyennes sur site pour montrer aux etudiants
	  l'invariance par translation des solutions et les effets des
	  degenerescences
	* faire implementer les correlations charge-chare

"""
import time
import numpy
from scipy.linalg import eigh
from scipy.sparse.linalg import eigs, eigsh
from matplotlib import pyplot as plt
import scipy.sparse as sps

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pandas as pd
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

#    print("Hilbert size : {0}".format(size))
    basis = numpy.arange(size)

    return basis, size


###########################################################
###########################################################
###########################################################
###########################################################


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

    data[eles] = field

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


###########################################################
###########################################################
###########################################################
###########################################################


def gen_hamiltonian_Ising(size, bond, basis, factors,
                          nx, ny):

    n_el = size + (size * nx * ny)

    row = numpy.zeros(n_el, dtype=int)
    column = numpy.zeros(n_el, dtype=int)
    data = numpy.zeros(n_el, dtype='double')

    echange, field = factors
    element_id = 0

#    print('Calculating bonds - field')

    row, column, data, element_id  = calculate_bonds_field(
            field, row, column, data, element_id,
            basis, bond)

#    print('Calculating bonds - Fneig interaction')

    (row, column,
     data, element_id) = calculate_bonds_f_neig_opt(
            echange, row, column, data, element_id,
            basis, bond)

#    print('Making sparse')

    ham_ech = sps.csc_matrix((data, (row, column)),
                             shape=(size, size))

    return ham_ech


###########################################################
###########################################################
###########################################################
###########################################################


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


###########################################################
###########################################################
###########################################################
###########################################################


def calculate_density_temp(basis, eigvect, ns, part_func,
                           eigener, T):

    density_vect_t = numpy.zeros(ns)

    for i, energy_i in enumerate(eigener):

        weigth = numpy.exp(-energy_i / T) / part_func

        density_vect = calculate_density(
                basis, eigvect[:, i][:, None],
                ns, show=False)

        density_vect_t = (density_vect_t
                          + (density_vect * weigth))

    return density_vect_t


def calculate_density(basis, eigvect, ns, show=True):


    denss = numpy.tile(
            (eigvect[:, 0]
             * numpy.conj(eigvect[:, 0]))[:, None],
            ns)

    mask = numpy.reshape(
            (numpy.bitwise_and(
                    numpy.repeat(basis, ns),
                    numpy.tile(
                            numpy.left_shift(
                                    1, numpy.arange(ns)),
                                 len(basis)))
             != 0),
            (len(basis), ns))

    density_vect = numpy.sum(denss * mask.astype(int), axis=0)

    if show:
        for i, dens in enumerate(density_vect):
            print("{0} \t {1}".format(i, dens))

    return density_vect


def calculate_mag_temp(basis, eigvect, ns, part_func,
                       eigener, T):

    mag_vect_t = numpy.zeros(ns)

    for i, energy_i in enumerate(eigener):

        weigth = numpy.exp(-energy_i / T) / part_func

        mag_vect = calculate_mag(
                basis, eigvect[:, i][:, None],
                ns, show=False)

        mag_vect_t = (mag_vect_t
                          + (mag_vect * weigth))

    return mag_vect_t


def calculate_mag(basis, eigvect, ns, show=True):


    ampl = numpy.tile(
            (eigvect[:, 0]
             * numpy.conj(eigvect[:, 0]))[:, None],
            ns)

    mask = numpy.reshape(
            (numpy.bitwise_and(
                    numpy.repeat(basis, ns),
                    numpy.tile(
                            numpy.left_shift(
                                    1, numpy.arange(ns)),
                                 len(basis)))
             != 0),
            (len(basis), ns))

    up = ampl * mask.astype(int)
    down = (ampl * numpy.logical_not(mask).astype(int))

    mag_vect = numpy.sum((up - down), axis=0)

    if show:
        for i, dens in enumerate(mag_vect):
            print("{0} \t {1}".format(i, dens))

    return mag_vect


# To optmize!!
def calculate_corr(basis, eigvect, ns,
                   mag_vect=None):

    moy_spin_i_j = numpy.zeros((ns, ns))
    for i in numpy.arange(ns):
        for j in numpy.arange(ns):
            corr = 0.0
            for w in numpy.arange(len(eigvect[:, 0])):
                is_up_i = (basis[w] & 1 << i) != 0
                is_up_j = (basis[w] & 1 << j) != 0

                if (is_up_i and is_up_j):
                    corr += (eigvect[w, 0]
                             * numpy.conj(eigvect[w, 0])) ** 2
                elif (is_up_i and (not is_up_j)):
                    corr += -(eigvect[w, 0]
                              * numpy.conj(eigvect[w, 0])) ** 2
                elif ((not is_up_i) and is_up_j):
                    corr += -(eigvect[w, 0]
                              * numpy.conj(eigvect[w, 0])) ** 2
                else:
                    corr += (eigvect[w, 0]
                             * numpy.conj(eigvect[w, 0])) ** 2

            moy_spin_i_j[i, j] = corr

    if mag_vect is None:
        mag_vect = calculate_mag(basis, eigvect,
                                 ns, show=False)[:, None]

    corre_matrix = (moy_spin_i_j
                    - numpy.transpose(mag_vect) * mag_vect)

    return corre_matrix


def solve(nx, ny, J, h, lattice='square', kvalues=1):

    ns = (nx * ny)

    basis, hamiltonian = create_cluster(
            nx, ny, factors=[J, h],
            lattice=lattice)

#    start = time.clock()

    #print("# ======== Diagonalization : brute force ")
    #EigenEnergies, EigenVectors = eigh(hamiltonian.todense())

#    print("# ======== Diagonalization : Lanczos")
    eigene, eigvect = eigs(
            hamiltonian, kvalues, None, None, 'SR', None,
            None, None, 1.e-5)

#    stop = time.clock()
#    print("{0} in {1} seconds".format(eigene[0],
#          stop-start))

    return eigene, eigvect, basis, ns


###########################################################
###########################################################
###########################################################
###########################################################


def cal_save(code, nJ, jmin, jmax,
             nh, hmin, hmax, kvalues, lattice, ny, nx,
             save_each_step=False):


    energy = numpy.zeros((nh,nJ))
    average_mag = numpy.zeros((nh,nJ))


    config_names = numpy.array(['nx', 'ny',
                                'lattice', 'kvalues'])
    config_values = numpy.array([nx, ny, 0, kvalues])
    h_values = numpy.linspace(hmin, hmax, nh)
    J_values = numpy.linspace(jmin, jmax, nJ)

    if save_each_step:
        size = [nh, 1]
    else:
        size = [nh, nJ]

    data = numpy.empty(size, dtype='object')

    for i, J in enumerate(J_values):
        for j, h in enumerate(h_values):
            eigene, eigvect, basis, ns = solve(
                    nx=4, ny=4, J=J, h=h, lattice=lattice,
                    kvalues=kvalues)

            mag_vect = calculate_mag(basis, eigvect,
                                     ns, show=False)

            data_i_j = {'eigene': eigene,
                        'eigvect': eigvect,
                        'basis': basis,
                        'ns': ns,
                        'mag_vect': mag_vect}

            if save_each_step:
                data[j, 0] = data_i_j
                average_mag[j, 0] = numpy.average(mag_vect)
                energy[j,0] = eigene[0]
            else:
                data[j, i] = data_i_j
                average_mag[j,i] = numpy.average(mag_vect)
                energy[j,i] = eigene[0]


        if save_each_step:

            data = data

            all_ = {'config_names': config_names,
                    'config_values': config_values,
                    'h_values': h_values,
                    'J_values': J_values,
                    'data': data,
                    'energy': energy,
                    'average_mag': average_mag}

            pd_frame = pd.DataFrame({'all': all_})

            pd_frame.to_pickle(
                    'config_ite_{0}_{1}_step_{2}'.format(nh * nJ, code, i))

        print('Done field {0} - {1}'.format(
                i, J))

    if not save_each_step:
        all_ = {'config_names': config_names,
                'config_values': config_values,
                'h_values': h_values,
                'J_values': J_values,
                'data': data,
                'energy': energy,
                'average_mag': average_mag}

        pd_frame = pd.DataFrame({'all': all_})

        pd_frame.to_pickle(
                'config_ite_{0}_{1}'.format(nh * nJ, code))

#    grid = numpy.column_stack((
#            numpy.repeat(numpy.arange(nJ), nh),
#            numpy.tile(numpy.arange(nh), nJ)))
#
#    extent = [hmin, hmax,
#              jmin*10, jmax*10]
#
#    plt.imshow(average_mag, vmin=average_mag.min(),
#               vmax=average_mag.max(), extent=extent,
#               cmap=cm.RdBu, interpolation='nearest')
#
#    plt.colorbar()
#    plt.show()
#
#    plt.imshow(energy, vmin=energy.min(),
#               vmax=energy.max(), extent=extent,
#               cmap=cm.RdBu, interpolation='nearest')
#
#    plt.colorbar()
#    plt.show()



###########################################################
###########################################################
###########################################################
###########################################################


nJ = 2
jmin = - 2
jmax = 2
nh = 2
hmin = -20
hmax = 20
kvalues=1
ny = 4
nx = 4


#code = 'T_1'
#lattice = 'triangular'
#
#cal_save(code, nJ, jmin, jmax,
#             nh, hmin, hmax, kvalues, lattice, ny, nx)
#
#print('Done T_1')

code = 'S_3'
lattice = 'square'

jmin = - 0.6
jmax = 0.0
hmin = -3
hmax = 3

cal_save(code, nJ, jmin, jmax,
             nh, hmin, hmax, kvalues, lattice, ny, nx,
             save_each_step=False)

print('Done S_2')

#code = 'S_K_100'
#kvalues = 100
#lattice = 'square'
#nJ = 20
#nh = 20
#
#cal_save(code, nJ, jmin, jmax,
#             nh, hmin, hmax, kvalues, lattice, ny, nx)
#
#print('Done S_K_100')

#code = 'S_K_200'
#kvalues = 200
#lattice = 'square'
#nJ = 20
#nh = 20
#
#cal_save(code, nJ, jmin, jmax,
#             nh, hmin, hmax, kvalues, lattice, ny, nx)
#
#print('Done S_K_200')

#grid = numpy.column_stack((
#        numpy.repeat(numpy.arange(nJ), nh),
#        numpy.tile(numpy.arange(nh), nJ)))
#
#extent = [hmin, hmax,
#          jmin*10, jmax*10]
#
#plt.imshow(average_mag, vmin=average_mag.min(),
#           vmax=average_mag.max(), extent=extent,
#           cmap=cm.RdBu, interpolation='nearest')
#
#plt.colorbar()
#plt.show()
#
#plt.imshow(energy, vmin=energy.min(),
#           vmax=energy.max(), extent=extent,
#           cmap=cm.RdBu, interpolation='nearest')
#
#plt.colorbar()
#plt.show()





#
## On met Kb = 1*
#nt=20
#mag_vs_T = numpy.zeros(nt)
#for T in range(nt):
#
#    if T is 0:
#        mag_vect = calculate_mag(basis, eigvect,
#                                 ns, show=False)
#    else:
#        part_func = numpy.sum(numpy.exp(-eigene / T))
#
#
#        mag_vect = calculate_mag_temp(basis, eigvect,
#                                        ns, part_func,
#                                        eigene, T=T)
#
#    mag_vs_T[T] = numpy.average(mag_vect)
#    print('Done T = {}'.format(T))
#
#plt.plot(numpy.arange(nt), mag_vs_T / mag_vs_T[0],
#         'o')
#plt.ylabel('Average magnetization')
#plt.xlabel('Temperature')
#plt.show()



#density_vect = calculate_density(basis, eigvect, ns,
#                                 show=False)
#density_vect_t = calculate_density_temp(basis, eigvect,
#                                        ns, part_func,
#                                        eigene, T=T)
#
#mag_vect = calculate_mag(basis, eigvect, ns, show=False)
#mag_vect_t = calculate_mag_temp(basis, eigvect,
#                                ns, part_func,
#                                eigene, T=T)
#
#print(density_vect_t)
#print(density_vect)
#
#print(mag_vect_t)
#print(mag_vect)
#


###########################################################
###########################################################
###########################################################
###########################################################


#corre_matrix = calculate_corr(basis, EigenVectors, ns,
#                              mag_vect=mag_vect[:, None])
#print(corre_matrix)
#print('Average : {}'.format(numpy.average(corre_matrix)))
#print('mag_tot = {}'.format(numpy.sum(mag_vect)))

## Correlation matrix
#corre_mat = numpy.ones((ns, ns)) * numpy.nan
#for s in numpy.arange(ns):
#    for w in numpy.arange(ns):
#        corre_mat[s, w] = density_vect[s] * density_vect[w]
#
#print(corre_mat)
#print(numpy.mean(corre_mat, axis=1))
