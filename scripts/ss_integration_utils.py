import celmech as cm
import numpy as np
from celmech.rk_integrator import RKIntegrator
from celmech.miscellaneous import _machine_eps
import pickle
_TOPDIR="/fs/lustre/cita/hadden/01_solar_system_secular/SolarSystemSecular/"
#_TOPDIR="/cita/h/home-2/hadden/Projects/05_SolarSystemSecular/"
_TOPDIR="/Users/shadden/Papers/43_SolarSystemSecular/"
_DEFAULT_SOURCE = _TOPDIR+"scripts/"
_DEFAULT_SOURCE += "secular_hamiltonian_4th_order.pkl"

_SEC_INFO_DIR =_TOPDIR+"synthetic_secular_soln/"
gvec = np.load(_SEC_INFO_DIR+"gvec.npy")
svec = np.load(_SEC_INFO_DIR+"svec.npy")
phase_e = np.load(_SEC_INFO_DIR+"phase_vec_e.npy")
phase_I = np.load(_SEC_INFO_DIR+"phase_vec_I.npy")
Smatrix_e = np.load(_SEC_INFO_DIR+"Secc_mtrx.npy")
Smatrix_I = np.load(_SEC_INFO_DIR+"Sinc_mtrx.npy")
Se_full = np.load(_SEC_INFO_DIR+"LL_matrix_e.npy")
SI_full = np.load(_SEC_INFO_DIR+"LL_matrix_I.npy")
L0s = np.load(_SEC_INFO_DIR+"L0s.npy")

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

def get_samples(n):
    """
    Generate a sample from joint distribution
    of n uniform random variables between
    0 and 1 subject to the constraint that
    their sum is equal to 1.
    """
    x = np.random.uniform(0,1,size=n-1)
    xs = np.sort(x)
    y = np.zeros(n)
    y[0] = xs[0]
    y[1:n-1] = xs[1:] - xs[:-1]
    y[n-1] = 1 - xs[-1]
    return y

def generate_inits(amd_e,amd_I,g1_frac=None,s1_frac=None):
    Se_terr = Se_full[:4,:4]
    SI_terr = SI_full[:4,:4]
    De,Te = np.linalg.eigh(Se_terr)
    DI,TI = np.linalg.eigh(SI_terr)
    phi = np.random.uniform(-np.pi,np.pi,size=8)
    psi = np.random.uniform(-np.pi,np.pi,size=8)
    psi[4] = phase_I[4] #
    I = np.zeros(4)
    if g1_frac:
        # g1 is the last mode
        I[-1] = g1_frac * amd_e
        I[:-1] = (1 - g1_frac) * get_samples(3) * amd_e
    else:
        I[:] = get_samples(4) * amd_e
    J = np.zeros(4)
    if s1_frac:
        J[0] = s1_frac * amd_e
        J[1:] = (1 - s1_frac) * get_samples(3) * amd_e
    else:
        J[:] = get_samples(4) * amd_e

    ufree = np.sqrt(I)*np.exp(1j * phi[:4])
    vfree = np.sqrt(J)*np.exp(1j * psi[:4])

    scale = np.diag(np.sqrt(0.5 * L0s[4:]))

    _ = np.transpose([Se_full[:4,4:] @ scale @ Smatrix_e[4:,i] for i in range(4,8)])
    bprime_e = np.transpose(Te.T @ _)
    uforced = np.sum([-1 * bprime_e[i] * np.exp(1j * phi[4 + i]) / (gvec[4 + i] + De) for i in range(2)],axis=0)

    _ = np.transpose([SI_full[:4,4:] @ (2*scale) @ Smatrix_I[4:,i] for i in range(4,8)])
    bprime_I = np.transpose(TI.T @ _)
    vforced = np.sum([-1 * bprime_I[i] * np.exp(1j * psi[4 + i]) / (svec[4 + i] + DI) for i in range(2)],axis=0)

    state = np.zeros(22)
    utot = ufree + uforced
    vtot = vfree + vforced
    xtot = Te @ utot
    ytot = TI @ vtot
    rt2 = np.sqrt(2)
    for i in range(4):
        state[2*i]      = -rt2 * np.imag(xtot[i])
        state[2*i+11]   =  rt2 * np.real(xtot[i])
        state[2*i+1]    = -rt2 * np.imag(ytot[i])
        state[2*i+1+11] =  rt2 * np.real(ytot[i])
    state[8]  = phi[4]
    state[9]  = phi[5]
    state[10] = psi[5]

    return state

def run_integration(ig,y0,Tfin,Nout,outfile="output"):
    Nsteps_between_out = int(np.ceil(Tfin/Nout/ig.dt))
    ytraj = np.zeros((Nout,y0.shape[0]))
    time = np.zeros(Nout)
    y=y0.copy()
    for i in xrange(Nout):
        ytraj[i] = y
        time[i] = i * Nsteps_between_out * ig.dt
        np.savez(outfile,trajectory=ytraj,time=time)
        for _ in xrange(Nsteps_between_out):
            y = ig.rk_step(y)

