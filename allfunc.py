import pyscf
import pathlib
from pyscf import tools
from pyscf import fci
import numpy as np
import scipy
from pyscf import ci
import pickle
import pyscf
from pyscf import mcscf
from pyscf import scf
from pyscf import gto
from pyscf import lo
import pathlib
from pyscf import tools
import pyscf.mcscf
from pyscf import mp
from pyscf import fci
from pyscf import ci
from pyscf.gto import Mole
from pyscf.scf import RHF
#from pyscf.mp.dfmp2_native import DFMP2
from pyscf import dft
from pyscf.gto import Mole
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    rdCoordGen,
)
import py3Dmol
from mpmath import chebyt, chop, taylor
from numpy.linalg import norm
from numpy import linalg as LA
import numpy as np
import scipy
from scipy import interpolate

def checkRDMs(ncas,nelecas,xyzfile,basis,spin,charge,verbose):
    #defining orbital extraction fn, using mean field
    def get_mo(mf, mol):
        """Get molecular orbitals"""
        #stores fock matrix orbs
        orbitals = {"canonical": mf.mo_coeff}

        # Get intrinsic bonding orbitals and localized intrinsic valence virtual orbitals (livvo)/ splits orbitals by occupation:
        orbocc = mf.mo_coeff[:, 0 : mol.nelec[0]]
        orbvirt = mf.mo_coeff[:, mol.nelec[0] :]

        #computes AO overlap matrix, not sure why
        ovlpS = mol.intor_symmetric("int1e_ovlp")

        #builds intrinsic atomic orbitals and makes them orthogonal
        iaos = lo.iao.iao(mol, orbocc)
        iaos = lo.orth.vec_lowdin(iaos, ovlpS)
        #computes intr. bond orbitals and stores them
        ibos = lo.ibo.ibo(mol, orbocc, locmethod="IBO")
        orbitals["ibo"] = ibos

        #computes Localized Intrinsic Virtual Valence Orbitals (what?)
        livvo = lo.vvo.livvo(mol, orbocc, orbvirt)
        orbitals["livvo"] = livvo
        return orbitals

    #directs the cube file writing
    def write_all_coeffs(
        mol, coeffs, prefix="cmo_"+str(xyzfile), dirname=".", margin=5, offset=0
    ):
        """Write cube files for the given coefficients."""
        #output exists?
        path = pathlib.Path(dirname)
        path.mkdir(parents=True, exist_ok=True)

        #loops throug MOs and writes the cube files
        for i in range(coeffs.shape[1]):
            outfile = f"{prefix}_{i+offset:02d}.cube"
            outfile = path / outfile
            print(f"Writing {outfile}")
            tools.cubegen.orbital(mol, outfile, coeffs[:, i], margin=margin)
    mol = gto.Mole()
    mol.atom = str(xyzfile)+".xyz"
    mol.charge=charge
    mol.spin=spin
    mol.verbose = 4
    mol.basis = basis
    mol.symmetry = True 
    mol.build()
    
    
    
    myhf = mol.RHF()
    myhf.chkfile = 'HF_'+str(xyzfile)+'.chk'
    myhf.init_guess = 'HF_'+str(xyzfile)+'.chk'
    myhf.max_cycle=5000
    myhf.kernel()

    orbitals=get_mo(myhf,mol)
    myhf.mo_energy.sort()
    d_hf={"Energy": myhf.mo_energy, "Occupancy": myhf.mo_occ}
    E_hf=myhf.e_tot
    mo_e_hf=myhf.mo_energy
    mo_occ_hf=myhf.mo_occ

        #Saving HF data
    with open('save_'+str(xyzfile)+'_HF.pkl', 'wb') as f:
        pickle.dump([E_hf,d_hf], f)
        write_all_coeffs(mol,orbitals["canonical"],prefix=f"HF_{xyzfile}",dirname="cmo_"+str(xyzfile),margin=5)

    mycas = myhf.CASSCF(ncas, nelecas)
    mycas.natorb=True
    mycas.chkfile='CAS_'+str(xyzfile)+'.chk'
    mycas.init_guess='CAS_'+str(xyzfile)+'.chk'
    mycas.verbose = 4
    mycas.kernel()

    casvec=mycas.ci
    #stores and visualizes final orbs
    orbitals=get_mo(mycas,mol)
    mycas.mo_energy.sort()
    d_cas={"Energy": mycas.mo_energy, "Occupancy": mycas.mo_occ}
    E_cas=mycas.e_tot
    mo_e_cas=mycas.mo_energy
    mo_occ_cas=mycas.mo_occ
    
        #Saving CASSCI data
    with open('save_'+str(xyzfile)+'_cas.pkl', 'wb') as f:
        pickle.dump([E_cas,d_cas], f)
    write_all_coeffs(mol,orbitals["canonical"],prefix=f"CAS_{xyzfile}",dirname="cmo_"+str(xyzfile),margin=5)


    # This entire snippet is defining the function make_Dmats that is called in the assembly of the G spin blocks. The required arguments are mol, defining the molecule, mycas, giving CASSCF orbitals, casvec, the CI vector, and ncas and nelecas, which define the active space.
    def make_Dmats(mol,mycas,casvec,ncas,nelecas):
    #From pyscf, the make_rdm12s function is used to get the D1-alpha and D1-beta blocks, then the D2-alpha-alpha, D2-alpha-beta, and D2-beta-beta geminal spin blocks.
        (dm1a,dm1b), (dm2aa,dm2ab,dm2bb) = mycas.fcisolver.make_rdm12s(casvec,ncas,nelecas)
    # nmo1g and nmo2g are empty arrays based on the the size/shape of the D1-alpha and D2-alpha-alpha spin blocks, respectively.  
        nmo1g=dm1a.shape[0]
        nmo2g=dm2aa.shape[0]
    #The PySCF outputs are indexed, or reindexed in the case of the 4D arrays.    
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
    #geminals are defined by combining indices i+j and k+l
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
    
    #The same process is applied to the other spin-blocks.
    
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
    (aD2aa,aD2ab,aD2bb),(D2aa,D2ab,D2bb), fullD2 = make_Dmats(mol,mycas,casvec,ncas,nelecas)
    def make_mGaaaa(mol,mycas,casvec,ncas,nelecas):  
    
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
  
        return mGaa,mGaa_2d 
    def make_mGabab(mol,mycas,casvec,ncas,nelecas):
        
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
                        print(f"m2Gabab[{i},{j},{k},{l}]={Gabab[i,j,k,l]}")
    
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
        return Gabab, Gabab_2d
    def make_mGaabb(mol,mycas,casvec,ncas,nelecas):
   
        (dm1a,dm1b), (dm2aa,dm2ab,dm2bb) = mycas.fcisolver.make_rdm12s(casvec,ncas,nelecas)
        
        
        nmo2g=dm2ab.shape[0]
        
        for i in range (nmo2g):
            for j in range (nmo2g):
                print(f"1D[{i},{j}]={dm1a[i,j]}") 
       
        Gaabb= np.zeros((nmo2g,nmo2g,nmo2g,nmo2g))
        mGaabb= np.zeros_like(Gaabb)
        mGaabb_2d = np.zeros((nmo2g**2,nmo2g**2))
        for i in range (nmo2g):
            for j in range (nmo2g):
                for k in range (nmo2g):
                    for l in range (nmo2g):
                        Gaabb[i,j,k,l]=aD2ab[i,l,j,k]
                        mGaabb[i,j,k,l]=Gaabb[i,j,k,l] - (dm1a[i,j]*dm1a[k,l])
                        print(f"m2Gaabb[{i},{j},{k},{l}]={mGaabb[i,j,k,l]}")
    
        exc_ab = []
        ind_ab = np.zeros((nmo2g, nmo2g), dtype='int32')
        count_ab = 0
    
        for i in range(nmo2g):
            for j in range(nmo2g):
                exc_ab.append([i, j])
                ind_ab[i, j] = count_ab
                count_ab += 1
    
        ab_size = nmo2g**2
        mGaabb_2d = np.zeros((ab_size, ab_size), dtype='float64')
    
        for ij in range(len(exc_ab)):
            i, j = exc_ab[ij]
            for kl in range(len(exc_ab)):
                k, l = exc_ab[kl]
                mGaabb_2d[ij, kl] = mGaabb[i, j, k, l]
   
        return mGaabb,mGaabb_2d
    
    print("\nMake mGaaaa...\n")
    Gaaaa,Gaaaa_2d=make_mGaaaa(mol,mycas,casvec,ncas,nelecas)
    print("\nMake mGabab and mGbaba...\n")
    Gabab,Gabab_2d=make_mGabab(mol,mycas,casvec,ncas,nelecas)
    Gbaba=np.transpose(Gabab_2d)
    print("\nMake mGaabb and mGbbaa...\n")
    Gaabb,Gaabb_2d=make_mGaabb(mol,mycas,casvec,ncas,nelecas)
    Gbbaa=np.transpose(Gaabb_2d)
    print("\nMake mGbbbb...\n")
    Gbbbb_2d=Gaaaa_2d.copy()

    print("\nCreating modified 2G matrix...\n")
    aBlockSize = Gaaaa_2d.shape[0]
    abBlockSize = Gabab_2d.shape[0]
    aabbBlockSize = Gaabb_2d.shape[0]
    bBlockSize = aBlockSize
    sizeG2= 2*aBlockSize + 2*abBlockSize
    fullG2= np.zeros((sizeG2,sizeG2),dtype='float64')
    fullG2[0:aBlockSize,0:aBlockSize]=Gaaaa_2d
    fullG2[0:aabbBlockSize,aBlockSize:aBlockSize+aabbBlockSize]=Gbbaa
    fullG2[aBlockSize:aBlockSize+aabbBlockSize,0:aabbBlockSize]=Gaabb_2d
    fullG2[aBlockSize:aBlockSize+aBlockSize,aBlockSize:aBlockSize+aBlockSize]=Gbbbb_2d
    fullG2[aBlockSize+aabbBlockSize:aBlockSize+aabbBlockSize+abBlockSize,aBlockSize+aabbBlockSize:aBlockSize+aabbBlockSize+abBlockSize]=Gabab_2d
    fullG2[(sizeG2-abBlockSize):sizeG2,(sizeG2-abBlockSize):sizeG2]=Gbaba
    print("\nCalculating Max λ_D and Max λ_G...\n")
    eigGs, vecGs=np.linalg.eigh(fullG2)
    eigDs, vecDs=np.linalg.eigh(fullD2)
    print("\nMax  λ_G:", max(eigGs))
    lamG=max(eigGs)
    if lamG > 1:
        print("\nExciton condensation detected!")
    else:
        print("\nNo Exciton condensation detected.")
        
    print("\nMax  λ_D:", max(eigDs))
    lamD= max(eigDs)
    if lamD > 1:
        print("\nCooper pair condensation detected!")
    else:
        print("\nNo Cooper pair condensation detected.")
    return (Gaaaa,Gaaaa_2d),(Gabab, Gabab_2d),(Gaabb, Gaabb_2d),(fullD2,fullG2), (lamD, lamG)



