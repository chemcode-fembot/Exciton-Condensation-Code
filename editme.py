from allfunc import checkRDMs
#leave this
#--------------------------------------------------------------------------------------#
###Necessary to do###

#choose the active space (n,n):
ncas, nelecas = (6,6)
#name of xyz file, within parent folder. make sure to put in quotes, without '.xyz'
xyzfile="water"
#basis (e.g. gaussian STO-nG)
basis='cc-pVDZ'
#spin value
spin=0
#charge
charge=0

#--------------------------------------------------------------------------------------#
#***you can leave these***#

#verbose tells pyscf how much information to give while running
verbose=4
#convergence tolerance tells the dE you want before SCF is considered converged. **Important** If not changing, set to "dfl" (needs quotes) or comment out the variable as #conv=...
conv="dfl" 
#max cycles limits the number of cycles CASSCF will run. **Important** If not changing, set to "dfl" (needs quotes) or comment out the variable as #max_cyc=...
max_cyc="dfl"

#leave this
(Gaaaa,Gaaaa_2d),(Gabab, Gabab_2d),(Gaabb, Gaabb_2d),(fullD2,fullG2), (lamD, lamG)=checkRDMs(ncas,nelecas,xyzfile,basis,spin,charge,verbose)

