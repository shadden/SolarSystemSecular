import numpy as np
import sys
from ss_integration_utils import initialize_integrator,generate_inits
from ss_integration_utils import run_integration

amd_frac_i = int(sys.argv[1])
file_id = int(sys.argv[2])

amd_nom = 2.5e-8
amd_fracs = np.linspace(0.1,1,9,endpoint=False)
amd_tot = amd_nom * amd_fracs[amd_frac_i]
ig = initialize_integrator(5000)
y0 = generate_inits(amd_tot,amd_tot)
#finame="./g1_0p01_s1_0p01_realization{}".format(file_id)

finame="./scaled_amd_sims/scaled_amd_val_{}_realization_{}".format(amd_frac_i,file_id)
Tfin =  5e9
Nout =  2**14
run_integration(ig,y0,Tfin,Nout,outfile=finame)
