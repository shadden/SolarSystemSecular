import numpy as np
import sys
from ss_integration_utils import initialize_integrator,generate_inits
from ss_integration_utils import run_integration

file_id = int(sys.argv[1])
amd_nom = 2.5e-8
ig = initialize_integrator(5000)
y0 = generate_inits(amd_nom,amd_nom,g1_frac = 0.01,s1_frac = 0.01)
finame="./g1_0p01_s1_0p01_realization{}".format(file_id)
Tfin =  5e9
Nout =  2**14
run_integration(ig,y0,Tfin,Nout,outfile=finame)
