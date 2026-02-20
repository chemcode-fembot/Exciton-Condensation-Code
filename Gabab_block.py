#Gabab_block.py


from Dmats import make_Dmats
import pyscf
import pathlib
from pyscf import tools
from pyscf import fci
import numpy as np
import scipy

def make_mGabab(mol,mycas,casvec,ncas,nelecas):
    
    (aD2aa,aD2ab,aD2bb),(D2aa,D2ab,D2bb), fullD2 = make_Dmats(mol,mycas,casvec,ncas,nelecas)
    #gmatrix 
    (dm1a,dm1b), (dm2aa,dm2ab,dm2bb) = mycas.fcisolver.make_rdm12s(casvec,ncas,nelecas)
    
    
    nmo2g=dm2ab.shape[0]
    
    for i in range (nmo2g):
        for j in range (nmo2g):
            print(f"1D[{i},{j}]={dm1a[i,j]}") 
   
    Gabab= np.zeros((nmo2g,nmo2g,nmo2g,nmo2g))
    Gabab_2d = np.zeros((nmo2g**2,nmo2g**2))
 
    for i in range (nmo2g):
        for j in range (nmo2g):
            for k in range (nmo2g):
                for l in range (nmo2g):
                    delta= 1 if j == l else 0
                    Gabab[i,j,k,l]=(delta* dm1a[i,k]) - aD2ab[i,l,k,j]
                    print(f"2Gabab[{i},{j},{k},{l}]={Gabab[i,j,k,l]}")

    exc_ab = []
    ind_ab = np.zeros((nmo2g, nmo2g), dtype='int32')
    count_ab = 0

    for i in range(nmo2g):
        for j in range(nmo2g):
            exc_ab.append([i, j])
            ind_ab[i, j] = count_ab
            count_ab += 1

    ab_size = nmo2g**2
    Gabab_2d = np.zeros((ab_size, ab_size), dtype='float64')

    for ij in range(len(exc_ab)):
        i, j = exc_ab[ij]
        for kl in range(len(exc_ab)):
            k, l = exc_ab[kl]
            Gabab_2d[ij, kl] = Gabab[i, j, k, l]


#    for i in range (nmo2g):
#        for j in range (nmo2g):
#            for k in range (nmo2g):
#                for l in range (nmo2g):
#                    ij= i*nmo2g + j
#                    kl= l*nmo2g + k
#                    Gabab_2d[ij,kl]  = Gabab[i,j,k,l]

    return Gabab, Gabab_2d
