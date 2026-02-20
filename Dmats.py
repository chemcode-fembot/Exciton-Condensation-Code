#Dmats.py



import pyscf
import pathlib
from pyscf import tools
from pyscf import fci
import numpy as np
import scipy
from pyscf import ci


def make_Dmats(mol,mycas,casvec,ncas,nelecas):
  
    (dm1a,dm1b), (dm2aa,dm2ab,dm2bb) = mycas.fcisolver.make_rdm12s(casvec,ncas,nelecas)
    nmo1g=dm1a.shape[0]
    nmo2g=dm2aa.shape[0]
    
    D1a= np.zeros((nmo1g,nmo1g))
    for i in range (nmo2g):
        for j in range (nmo2g):
            D1a[i,j]=dm1a[i,j]

    aD2aa= np.zeros((nmo2g,nmo2g,nmo2g,nmo2g))
    D2aa= np.zeros((nmo2g**2,nmo2g**2))
    for i in range (nmo2g):
        for j in range (nmo2g): #i < j
            for k in range (nmo2g):
                for l in range (nmo2g): #k < l
                    aD2aa[i,j,k,l]=dm2aa[i,k,j,l]

    gems_aa = []
    ind_aa = np.zeros((nmo2g,nmo2g), dtype='int32')
    count_aa = 0
    for i in range(0,nmo2g):
        for j in range(0,nmo2g):
            if i < j:
                gems_aa.append([i,j])
                ind_aa[i,j] = count_aa
                ind_aa[j,i] = count_aa
                count_aa += 1
            elif i == j:
                ind_aa[i,j] = -999
    for ij in range(0,len(gems_aa)):
        i = gems_aa[ij][0]
        j = gems_aa[ij][1]
        for kl in range(0,len(gems_aa)):
            k=gems_aa[kl][0]
            l=gems_aa[kl][1]
            D2aa[ij,kl]=aD2aa[i,j,k,l]



    aD2ab= np.zeros((nmo2g,nmo2g,nmo2g,nmo2g))
    D2ab= np.zeros((nmo2g**2,nmo2g**2))
    for i in range (nmo2g):
        for j in range (nmo2g):
            for k in range (nmo2g):
                for l in range (nmo2g):  
                    aD2ab[i,j,k,l]=dm2ab[i,k,j,l]
    gems_ab = []
    ind_ab = np.zeros((nmo2g,nmo2g), dtype='int32')
    count_ab = 0
    for i in range(0,nmo2g):
        for j in range(0,nmo2g):
                gems_ab.append([i,j])
                ind_ab[i,j] = count_ab
                count_ab += 1
         
    for ij in range(0,len(gems_aa)):
        i = gems_ab[ij][0]
        j = gems_ab[ij][1]
        for kl in range(0,len(gems_aa)):
            k=gems_ab[kl][0]
            l=gems_ab[kl][1]
            D2ab[ij,kl]=aD2ab[i,j,k,l]
    
    aD2bb = np.zeros((nmo2g,nmo2g,nmo2g,nmo2g))
    D2bb= np.zeros((nmo2g**2,nmo2g**2))
    for i in range (nmo2g):
        for j in range (nmo2g):
            for k in range (nmo2g):
                for l in range (nmo2g):
                    aD2bb[i,j,k,l]=dm2bb[i,k,j,l]           
    gems_bb = []
    ind_bb = np.zeros((nmo2g,nmo2g), dtype='int32')
    count_bb = 0
    for i in range(0,nmo2g):
        for j in range(0,nmo2g):
            if i < j:
                gems_bb.append([i,j])
                ind_bb[i,j] = count_bb
                ind_bb[j,i] = count_bb
                count_bb += 1
            elif i == j:
                ind_bb[i,j] = -999
    for ij in range(0,len(gems_bb)):
        i = gems_bb[ij][0]
        j = gems_bb[ij][1]
        for kl in range(0,len(gems_bb)):
            k=gems_bb[kl][0]
            l=gems_bb[kl][1]
            D2bb[ij,kl]=aD2bb[i,j,k,l]

    #assembly:
    aaSize= D2aa.shape[0]
    sizeD2= 2*aaSize + nmo2g**2
    fullD2= np.zeros((sizeD2,sizeD2),dtype='float64')
    fullD2[0:aaSize,0:aaSize]=D2aa
    fullD2[aaSize:aaSize+nmo2g**2,aaSize:aaSize+nmo2g**2]=D2ab
    fullD2[aaSize+nmo2g**2:sizeD2,aaSize+nmo2g**2:sizeD2]=D2bb

    return (aD2aa,aD2ab,aD2bb),(D2aa,D2ab,D2bb), fullD2

