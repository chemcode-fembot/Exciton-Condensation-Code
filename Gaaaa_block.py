#Gaaaa_block.py

from Dmats import make_Dmats
import pyscf
import pathlib
from pyscf import tools
from pyscf import fci
import numpy as np
import scipy
from pyscf import ci

def make_mGaaaa(mol,mycas,casvec,ncas,nelecas):  

    (aD2aa,aD2ab,aD2bb),(D2aa,D2ab,D2bb), fullD2 = make_Dmats(mol,mycas,casvec,ncas,nelecas)
    eigDs, vecDs=np.linalg.eigh(fullD2)
    #gmatrix 
    (dm1a,dm1b), (dm2aa,dm2ab,dm2bb) = mycas.fcisolver.make_rdm12s(casvec,ncas,nelecas)
    #d2aa_corrected = dm2aa.transpose(0, 2, 1, 3)
    nmo2g=dm2aa.shape[0]
    
    for i in range (nmo2g):
        for j in range (nmo2g):
            print(f"1D[{i},{j}]={dm1a[i,j]}")

    Gaa= np.zeros((nmo2g,nmo2g,nmo2g,nmo2g))
    mGaa = np.zeros_like(Gaa)
    mGaa_2d = np.zeros((nmo2g**2,nmo2g**2))
    for i in range (nmo2g):
        for j in range (nmo2g):
            for k in range (nmo2g):
                for l in range (nmo2g):
                    delta= 1 if j == l else 0
                    Gaa[i,j,k,l]=(delta* dm1a[i,k]) - aD2aa[i,l,k,j]
                    mGaa[i,j,k,l]=Gaa[i,j,k,l] - (dm1a[i,j]*dm1a[k,l])
                    print(f"m2Gaa[{i},{j},{k},{l}]={mGaa[i,j,k,l]}")

    
    exc_aa = []
    ind_aa = np.zeros((nmo2g, nmo2g), dtype='int32')
    count_aa = 0

    for i in range(nmo2g):
        for j in range(nmo2g):
                exc_aa.append([i, j])
                ind_aa[i, j] = count_aa
                ind_aa[j, i] = count_aa
                count_aa += 1

    aa_size = int(nmo2g**2)
    mGaa_2d = np.zeros((aa_size, aa_size), dtype='float64')

    for ij in range(len(exc_aa)):
        i, j = exc_aa[ij]
        for kl in range(len(exc_aa)):
            k, l = exc_aa[kl]
            mGaa_2d[ij, kl] = mGaa[i, j, k, l]
            print(f"m2Gaa2d[{ij},{kl}]={mGaa_2d[ij,kl]}")

#    for i in range (nmo2g):
#        for j in range (nmo2g):
#            for k in range (nmo2g):
#                for l in range (nmo2g):
#                    ij= i*nmo2g + j
#                    kl= k*nmo2g + l
#                    mGaa_2d[ij,kl]  = mGaa[i,j,k,l]


    return mGaa,mGaa_2d,eigDs

