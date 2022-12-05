import celmech as cm
from celmech.rk_integrator import RKIntegrator
from celmech.miscellaneous import _machine_eps
import pickle
_DEFAULT_SOURCE =  "/home/hadden/Projects/SolarSystemSecular/scripts/"
_DEFAULT_SOURCE += "secular_hamiltonian_4th_order.pkl"
def initialize_integrator(dt):
    ham = initialize_hamiltonian()
    def flow(y):
        return ham.flow_func(*y).reshape(-1)
    def flow_and_jac(y):
        return flow(y), ham.jacobian_func(*y)
    integrator = RKIntegrator(
        flow,
        flow_and_jac,
        ham.N_dim,
        dt,
        rtol=_machine_eps,
        atol=1e-12,
        rk_method='GL6',
        rk_root_method='Newton',
        max_iter=10
    )
    return integrator

def initialize_hamiltonian(source_file = _DEFAULT_SOURCE):
    with open(source_file,"rb") as fi:
        htot,state = pickle.load(fi)
    ham = cm.Hamiltonian(htot,{},state)
    return ham

