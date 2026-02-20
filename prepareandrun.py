#prepareandrun.py


#imports from the G files
from Gaaaa_block import make_mGaaaa
from Gabab_block import make_mGabab
from Gaabb_block import make_mGaabb
#for saving
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
    mol, coeffs, prefix="cmo", dirname=".", margin=5, offset=0
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


#Defining the molecule
mol = gto.Mole()
mol.atom = "coronene.xyz"
mol.charge=0
mol.spin=0
mol.verbose = 4
mol.basis = 'STO3G' #These are the basis functions that we are using.

mol.symmetry = True #Allowing the function to exploit symmetry (think inorganic chemistry)

mol.build()



#running a base HF calculation
myhf = mol.RHF()
myhf.chkfile = 'HF.chk'
myhf.init_guess = 'HF.chk'
myhf.max_cycle=100
myhf.kernel()
#gets canonical*? and localized orbs, sorts MO energies and stores with occupations

orbitals=get_mo(myhf,mol)
myhf.mo_energy.sort()
d_hf={"Energy": myhf.mo_energy, "Occupancy": myhf.mo_occ}
E_hf=myhf.e_tot
mo_e_hf=myhf.mo_energy
mo_occ_hf=myhf.mo_occ


#Saving HF data
with open('save_coronene_HF.pkl', 'wb') as f:
    pickle.dump([E_hf,d_hf], f)
write_all_coeffs(mol,orbitals["canonical"],prefix=f"HF_cor",dirname="cmo",margin=5)

#running the CASSCF (Complete Active Space SCF) calculation (using 4,4 for speed now) and uses checkpointing
ncas, nelecas = (10,10)
mycas = myhf.CASSCF(ncas, nelecas)
mycas.natorb=True
mycas.chkfile='CAS.chk'
mycas.init_guess='CAS.chk'
mycas.verbose = 4
mycas.kernel()


#extracts CI vector
casvec=mycas.ci


#stores and visualizes final casscf orbs
orbitals=get_mo(mycas,mol)
mycas.mo_energy.sort()
d_cas={"Energy": mycas.mo_energy, "Occupancy": mycas.mo_occ}
E_cas=mycas.e_tot
mo_e_cas=mycas.mo_energy
mo_occ_cas=mycas.mo_occ


#Saving CASSCF data
with open('save_coronene_cas.pkl', 'wb') as f:
    pickle.dump([E_cas,d_cas], f)
write_all_coeffs(mol,orbitals["canonical"],prefix=f"CAS_cor",dirname="cmo",margin=5)


#run Gaaaa
print("\nMake mGaaaa...\n")
Gaaaa,Gaaaa_2d,eigDs=make_mGaaaa(mol,mycas,casvec,ncas,nelecas)
#run Gabab
print("\nMake mGabab and mGbaba...\n")
Gabab,Gabab_2d=make_mGabab(mol,mycas,casvec,ncas,nelecas)
Gbaba=np.transpose(Gabab_2d)
#run Gaabb
print("\nMake mGaabb and mGbbaa...\n")
Gaabb,Gaabb_2d=make_mGaabb(mol,mycas,casvec,ncas,nelecas)
Gbbaa=np.transpose(Gaabb_2d)
#run Gbbbb
print("\nMake mGbbbb...\n")
Gbbbb_2d=Gaaaa_2d.copy()

print("\nMax  λ_D:", max(eigDs))


#G assembly
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
print("\nCalculating Max  λ_G...\n")
eigGs, vecGs=np.linalg.eigh(fullG2)
print("\nMax  λ_G:", max(eigGs))



