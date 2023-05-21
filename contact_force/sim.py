import numpy as np
from scipy.special import ellipk
import pandas as pd
import torch
import time as time_
from contact_force.funcs import hertzian
from scipy.integrate import cumtrapz
from scipy.interpolate import splrep, splev

def In_L(r_n, dr):
    '''
    using: Interaction and Deformation of Elastic Bodies: Origin of Adhesion Hysteresis - Phil Attard
    calculate the left hand limit at the singularity in the complete elliptic integral of the first kind
    :param r_n: (float) singularity point
    :param dr: (float) discretization
    :return: (float) analytically evaluated left hand limit
    '''
    return ((dr / np.pi - dr ** 2 / (4 * np.pi * r_n)) *
            (1 + np.log(16 * r_n ** 2 / (r_n * dr - dr ** 2 / 4))))


def In_R(r_n, dr):
    '''
    using: Interaction and Deformation of Elastic Bodies: Origin of Adhesion Hysteresis - Phil Attard
    calculate the right hand limit at the singularity in the complete elliptic integral of the first kind
    :param r_n: (float) singularity point
    :param dr: (float) discretization
    :return: (float) analytically evaluated right hand limit
    '''
    return (dr / np.pi * np.log(16 * (r_n + dr / 2) ** 2 / (r_n * dr + dr ** 2 / 4)) +
            4 * r_n / np.pi * np.log((2 * r_n + dr) / (2 * r_n + dr / 2)))


def K(r, r_p, dr):
    '''
    using: Interaction and Deformation of Elastic Bodies: Origin of Adhesion Hysteresis - Phil Attard
    calculate the complete elliptic integral of the first kind for handling the radial integrating kernel
    :param r: (float) radius value
    :param r_p: (numpy vector (nx1)) dummy radius axis
    :param dr: (float) radius axis (including dummy axis) discretization
    :return: (numpy matrix (nxn)) analytically evaluated complete elliptic integral of the first kind
    '''
    values = np.zeros(r_p.shape)  # make matrix
    values[r_p < r] = ellipk(r_p[r_p < r] ** 2 / r ** 2) * 4 / (r * np.pi)  # handle up to singularity
    values[r_p > r] = ellipk(r ** 2 / r_p[r_p > r] ** 2) * 4 / (r_p[r_p > r] * np.pi)  # handle past singularity
    values[r_p == r] = (In_L(r, dr) + In_R(r, dr)) / (r * dr)  # analytically evaluate singularity
    return values


def f_sphere(r, R):
    '''
    calculate the radially parameterized position of the contacting indenter
    :param r: (numpy array) radius axis
    :param R: (float) radius of probe
    :return: (numpy array) position of the contacting indenter
    '''
    return r ** 2 / (2 * R)


def h_calc(h0, u, f_tip, r, *args):
    '''
    calculate the radially parameterized separation between the indenter and the surface
    :param h0: (float) 'nominal separation' (in the language of Attard) absolute position of indenter relative to
    the undeformed sample surface
    :param u: (numpy array) radially parameterized deformation of the surface
    :param f_tip: function of the form f_tip(r, *args) which parameterizes the probe position
    :param r: (numpy array) radial discretization
    :param args: (optional) arguments of the f_tip probe position calculation function
    :return: (numpy array) distance between the indenter and the surface
    '''
    return h0 + f_tip(r, *args) - u


def p_vdw(h, H1=1e-19, H2=1e-19, z0=1e-9):
    '''
    calculates the van der Waals pressure and derivative as a function of distance
    :param h: (float or numpy array) distance between bodies
    :param H1: (float) repulsive Hamaker constant
    :param H2: (float) attractive Hamaker constant
    :param z0: (float) equilibrium separation distance
    :return: (float or numpy array x2) pressure, derivative of pressure with respect to distance
    '''
    return (1 / (6 * np.pi * h ** 3) * (H1 * (z0 / h) ** 6 - H2),
            1 / (2 * np.pi * h ** 4) * (H2 - 3 * H1 * (z0 / h) ** 6))

def p_hw(h, H=1e-19, z0=1e-9, n=9):
    '''
    calculates the physical apprximation of the hard-wall repulsive pressure and derivative as a function of distance
    :param h: (float or numpy array) distance between bodies
    :param H: (float) repulsive Hamaker constant
    :param z0: (float) equilibrium separation distance
    :param n: (int) repulsion exponent
    :return: (float or numpy array x2) pressure, derivative of pressure with respect to distance
    '''
    return (H * z0 ** (n - 3) / (6 * np.pi * h ** n),
            - n * H * z0 ** (n - 3) / (6 * np.pi * h ** n))

def p_exp_approx(h, H_rep=1e7, k_inv=1/1e-9, h_off=0):
    '''
    calculates an approximation for an exponential repulsion potential
    :param h: float distance between bodies
    :param H_rep: float repulsive Hamaker constant
    :param k_inv: float debye length
    :param h_off: float offset shift to distance
    :return: float pressure and pressure derivative
    '''
    # P, Ph
    return (H_rep / (h_off + k_inv * h),
            - H_rep * k_inv / (h_off + k_inv * h) ** 2)

def p_exp(h, magnitude=1e7, decay_length=1e-9):
    '''
    calculates an exponential repulsive potential distribution (cannot generally be used on its own)
    :param h: float distance between bodies
    :param magnitude: float magnitude of repulsive force at 0 separation
    :param decay_length: float decay constant
    :return: float pressure and pressure derivative
    '''
    return (magnitude * np.exp(- h / decay_length),
            - magnitude / decay_length * np.exp(- h / decay_length))

def p_exp_TORCH(h, magnitude=1e7, decay_length=1e-9):
    '''
    calculates an exponential repulsive potential distribution (cannot generally be used on its own)
    FOR PYTORCH TENSORS
    :param h: float distance between bodies
    :param magnitude: float magnitude of repulsive force at 0 separation
    :param decay_length: float decay constant
    :return: float pressure and pressure derivative
    '''
    return (magnitude * torch.exp(- h / decay_length),
            - magnitude / decay_length * torch.exp(- h / decay_length))
def p_exp_and_rep(h, magnitude=1e7, decay_length=1e-9, H_rep=1e-19, z0=0.5e-9):
    '''
    superposition of exponential repulsion and short range, hard wall repulsion
    :param h: float distance between bodies
    :param magnitude: float magnitude of exponential repulsive force at 0 separation
    :param decay_length: float decay constant of exponential force
    :param H_rep: float hard wall repulsion hamaker constant
    :param z0: decay constant of hard wall repulsion
    :return: float pressure and pressure derivative
    '''
    p1, ph1 = p_exp(h, magnitude, decay_length)
    p2, ph2 = p_vdw(h, H2=0, H1=H_rep, z0=z0)
    return (p1 + p2,
            ph1 + ph2)

def p_exp_and_rep_TORCH(h, magnitude=1e7, decay_length=1e-9, H_rep=1e-19, z0=0.5e-9):
    '''
    superposition of exponential repulsion and short range, hard wall repulsion
    FOR PYTORCH TENSORS
    :param h: float distance between bodies
    :param magnitude: float magnitude of exponential repulsive force at 0 separation
    :param decay_length: float decay constant of exponential force
    :param H_rep: float hard wall repulsion hamaker constant
    :param z0: decay constant of hard wall repulsion
    :return: float pressure and pressure derivative
    '''
    p1, ph1 = p_exp_TORCH(h, magnitude, decay_length)
    p2, ph2 = p_vdw(h, H2=0, H1=H_rep, z0=z0)
    return (p1 + p2,
            ph1 + ph2)


def p_degennes(h, L0=2e-6, N=3e14, T=300, H_rep=1e-19, z0=0.5e-9):
    '''
    de-gennes grafted polymer repulsive pressure superposed with a hard-wall repulsion
    coefficients taken from sokolov (experiments on cells): https://doi.org/10.1063/1.2757104
    model derived using Surface and Interaction Forces 2ed (pp. 341)
    :param h: float separation between ungrafted surface and grafted surface
    :param L0: float polymer equilibrium length (sokolov reports 2 microns on average)
    :param N: float polymer grafting density (sokolov reports about 300 polymers per square micron)
    :param T: float temperature
    :param H_rep: float hard-wall repulsion magnitude
    :param z0: float hard-wall repulsion distance
    :return: (float, float) pressure and pressure derivative
    '''
    kb = 1.38e-23
    C = kb * T * L0 * N ** (3 / 2)

    P = 2 * C * (L0 ** (5 / 4) * h ** (-9 / 4) - L0 ** (-7 / 4) * h ** (3 / 4))
    Ph = - C / 2 * (3 * L0 ** (-7 / 4) * h ** (-1 / 4) + 9 * L0 ** (5 / 4) * h ** (-13 / 4))

    p_hw, ph_hw = p_vdw(h, H1=H_rep, H2=0, z0=z0)
    return (P + p_hw, Ph + ph_hw)

def rk4(state, dt, rhs_func, *args):
    '''
    Runge Kutta 4th order time integration update scheme
    :param state: (numpy array) variables to integrate
    :param dt: (float) time discretization
    :param rhs_func: function to calculate the time derivatives of the state of form rhs_func(state, *args)
    returns (time derivative of state vector, (extra stuff))
    :param args: (optional) arguments for the rhs_func
    :return: (numpy array) integrated state at t=t+dt, (extra stuff calculated at final step)
    '''
    k1, _ = rhs_func(state, *args)
    k2, _ = rhs_func(state + dt * k1 / 2, *args)
    k3, _ = rhs_func(state + dt * k2 / 2, *args)
    k4, extra = rhs_func(state + dt * k3, *args)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6, extra

def rhs_N_1_rigid(state, r, dr, R, k_ij, I, v0, b1, b0, c1, c0, p_func, *args):
    '''
    calculates the derivative of the state vector (surface deformation and probe position) assuming that the
    probe position is prescribed by a time dependent velocity v0(t) (i.e. by a rigid cantilever)
    also assuming that the viscoelastic ODE is order N=1
    :param state: numpy array state vector (deformation, probe position)
    :param r: float radial spatial axis
    :param dr: float spatial discretization
    :param R: float spherical probe radius
    :param k_ij: numpy matrix (nr x nr) integrating kernel
    :param I: numpy matrix (nr x nr) identity matrix
    :param v0: float time dependent prescribed velocity of the probe
    :param b1: float viscoelastic coefficient b1
    :param b0: float viscoelastic coefficient b0
    :param c1: float viscoelastic coefficient c1
    :param c0: float viscoelastic coefficient c0
    :param p_func: function for calculating sphere-point pressure distribution p_func(h, *args)
    :param args: optional arguments for p_func
    :return: numpy array derivative of state vector (deformation rate, probe velocity)
    '''
    u, h0 = state  # unpack the state vector
    h = h_calc(h0, u, f_sphere, r, R)  # calculate the separation
    p, p_h = p_func(h, *args)  # calculate the pressure and derivative
    # calculate LHS G matrix
    G_ij = c1 * dr * k_ij * p_h * r - b1 * I
    # calculate RHS vectors, d_j
    d_j = c1 * v0 * dr * (p_h * r) @ k_ij + c0 * dr * (p * r) @ k_ij + b0 * u
    u_dot = np.linalg.solve(G_ij, d_j)  # solve the system of equations
    return np.array([u_dot, v0], dtype='object'), (p, h)

def rhs_N_1_rigid_cuda(state, r, dr, R, k_ij, I, v0, b1, b0, c1, c0, p_func, *args):
    '''
    calculates the derivative of the state vector (surface deformation and probe position) assuming that the
    probe position is prescribed by a time dependent velocity v0(t) (i.e. by a rigid cantilever)
    also assuming that the viscoelastic ODE is order N=1
    ASSUMES VARIABLES ARE TORCH TENSORS
    :param state: numpy array state vector (deformation, probe position)
    :param r: float radial spatial axis
    :param dr: float spatial discretization
    :param R: float spherical probe radius
    :param k_ij: numpy matrix (nr x nr) integrating kernel
    :param I: numpy matrix (nr x nr) identity matrix
    :param v0: float time dependent prescribed velocity of the probe
    :param b1: float viscoelastic coefficient b1
    :param b0: float viscoelastic coefficient b0
    :param c1: float viscoelastic coefficient c1
    :param c0: float viscoelastic coefficient c0
    :param p_func: function for calculating sphere-point pressure distribution p_func(h, *args)
    :param args: optional arguments for p_func
    :return: numpy array derivative of state vector (deformation rate, probe velocity)
    '''
    u = state[:, 0]  # unpack the state vector
    h0 = state[:, 1]  # unpack the state vector
    h = h_calc(h0, u, f_sphere, r, R)  # calculate the separation
    p, p_h = p_func(h, *args)  # calculate the pressure and derivative
    # calculate LHS G matrix
    G_ij = c1 * dr * k_ij * p_h * r - b1 * I
    # calculate RHS vectors, d_j
    d_j = c1 * v0 * dr * k_ij @ (p_h * r) + c0 * dr * k_ij @ (p * r) + b0 * u
    u_dot = torch.linalg.solve(G_ij, d_j)  # solve the system of equations
    return torch.cat((torch.reshape(u_dot, v0.shape), v0), 1), (p, h)


def rhs_N_1_cantilever_cuda(u, zb, zt, vt, t, f_exc_func, vb_func, r, dr, R, k_ij, I, b1, b0, c1, c0, m, w0, Q0, K,
                            p_func, *args):
    '''
    calculates the derivative of the state vector (surface deformation, base position, tip position, tip velocity)
    assuming that the tip position is governed by a damped harmonic oscillator subjected to an excitation force
    with a moveable base and an intermittent, long-range interaction with a linear (visco)elastic material
    also assuming that the viscoelastic ODE is order N=1
    :param u: torch float tensor displacement of surface (positive upwards) (m)
    :param zb: float position of base (m)
    :param zt: float position of tip (m)
    :param vt: float velocity of tip (m/s)
    :param t: float time (s)
    :param f_exc_func: function that takes arguments (base pos, tip-sample force, tip pos, tip vel, and time) which
    can be written for specific simulation runs - i.e. one may desire a resonant tapping mode something more complicated
    where the excitation force changes based off of typical AFM observables which are given as arguments
    :param vb_func: function that takes same arguments as f_exc_func (same philosophy here) that calculates the velocity
    of the cantilever base
    :param r: torch float tensor radial spatial axis (m)
    :param dr: float spatial discretization (m)
    :param R: float spherical probe radius (m)
    :param k_ij: numpy matrix (nr x nr) integrating kernel
    :param I: numpy matrix (nr x nr) identity matrix
    :param b1: float viscoelastic coefficient b1
    :param b0: float viscoelastic coefficient b0
    :param c1: float viscoelastic coefficient c1
    :param c0: float viscoelastic coefficient c0
    :param p_func: function for calculating sphere-point pressure distribution p_func(h, *args)
    :param args: optional arguments for p_func
    :return: derivative of state variables: (deformation rate of surface (torch float tensor), base velocity (float),
    tip velocity (float), tip acceleration (float)) + extra useful information (f_ts, f_exc, p, h)
    '''
    h = h_calc(zt, u, f_sphere, r, R)  # calculate the separation
    p, p_h = p_func(h, *args)  # calculate the pressure and derivative
    f_ts = 2 * np.pi * p @ r * dr  # calculate the tip-sample force
    # calculate LHS G matrix
    G_ij = c1 * dr * k_ij * p_h * r - b1 * I
    # calculate RHS vectors, d_j
    d_j = c1 * vt * dr * k_ij @ (p_h * r) + c0 * dr * k_ij @ (p * r) + b0 * u
    u_dot = torch.linalg.solve(G_ij, d_j)  # solve the system of equations
    vb = vb_func(zb, f_ts, zt, vt, t)
    f_exc = f_exc_func(zb, f_ts, zt, vt, t)
    at = 1 / m * (f_ts + f_exc + K * zb - m * w0 / Q0 * vt - K * zt)
    return (u_dot, vb, vt, at), (f_ts, f_exc, p, h)

def get_coefs_rajabifar(Gg, Ge, Tau, v):
    '''
    get viscoelastic ODE coefficients in the style of Rajabifar and Attard
    :param Gg: float instantaneous shear modulus
    :param Ge: float equilibrium shear modulus
    :param Tau: float relaxation time
    :param v: float poisson's ratio
    :return: b1, b0, c1, c0
    '''
    return 1, 1 / Tau, (1 - v ** 2) / Gg, (1 - v ** 2) / (Ge * Tau)

def get_coefs_sls(Gg, Ge, Tau, v):
    '''
    get viscoelastic ODE coefficients for a standard linear solid (SLS) material
    :param Gg: flaot instantaneous shear modulus
    :param Ge: float equilibrium shear modulus
    :param Tau: float relaxation time
    :param v: float poisson's ratio
    :return: b1, b0, c1, c0
    '''
    return Gg * Tau, Ge, (1 - v ** 2) * Tau, (1 - v ** 2)

def get_coefs_elastic(G, v):
    '''
    get viscoelastic ODE coefficients for an elastic solid
    :param G: float shear modulus
    :param v: float poisson's ratio
    :return: b1, b0, c1, c0
    '''
    return G, 0, (1 - v ** 2), 0

def get_coefs_viscous(mu, v):
    '''
    get viscoelastic ODE coefficients for a viscous solid
    :param mu: float viscosity
    :param v: float poisson's ratio
    :return: b1, b0, c1, c0
    '''
    return mu, 0, 0, (1 - v ** 2)

def get_coefs_kvf(J, Tau, v):
    '''
    get viscoelastic ODE coefficients for a kelvin-voigt viscous solid (fluid)
    :param J: float shear compliance
    :param Tau: float retardation time
    :param v: float poisson's ratio
    :return: b1, b0, c1, c0
    '''
    return Tau, 1, 0, (1 - v ** 2) * J

def get_coefs_mf(G, Tau, v):
    '''
    get viscoelastic ODE coefficients for a maxwell viscous solid (fluid)
    :param G: float shear modulus
    :param Tau: float relaxation time
    :param v: float poisson's ratio
    :return: b1, b0, c1, c0
    '''
    return G * Tau, 0, (1 - v ** 2) * Tau, (1 - v ** 2)

def get_coefs_gkv(Jg, Je, Tau, v):
    '''
    get viscoelastic ODE coefficients for (generalized) kelvin-voigt solid
    :param Jg: float instantaneous shear compliance
    :param Je: float equilibrium shear compliance
    :param Tau: float retardation time
    :param v: float poisson's ratio
    :return: b1, b0, c1, c0
    '''
    return Tau, 1, (1 - v ** 2) * Jg * Tau, (1 - v ** 2) * Je


def get_explicit_sim_arguments(l, f, rho_c, R, G_star, N_R, H_R_target=-0.1, F_target=1):
    '''
    calculate the explicit arguments for simulate_rigid_N1 given the dimensionless quantities l and f
    using the dimension of R (probe radius), G_star (reduced shear modulus)
    :param l: float (1e-2 to 1e3, theoretically also to inf) dimensionless length scale l=(Rh')^1/2
    :param f: float (theoretically from 0, 1e-10 to 1e6) dimensionless pressure scale f=P0/G*
    :param rho_c: float (usually 1 to 10) dimensionless maximum radial distance in discrete domain
    :param R: float probe radius in m
    :param G_star: float reduced shear modulus in Pa (G*=G/(1-v^2))
    :param N_R: int number of points in discrete domain
    :param H_R_target: float (-inf, inf) target probe position as a ratio of probe radius
    :param F_target: float (0, inf) ratio of theoretical hertz force obtained at maximum indentation (usually >1)
    :return:
    '''
    r_c = rho_c * R / l
    d_r = r_c / N_R
    P_magnitude = f * G_star
    decay_constant = R / l ** 2
    pos_target = R * H_R_target
    # estimate the maximum force as a ratio of the maximum hertzian force
    force_target = abs(pos_target) ** 1.5 * G_star * R ** 0.5 * F_target
    return r_c, d_r, P_magnitude, decay_constant, pos_target, force_target



def simulate_rigid_N1(Gg, Ge, Tau, v, v0, h0, R, p_func, *args,
                      nr=int(1e3), dr=1.5e-9,
                      dt=1e-4, nt=int(1e6),
                      force_target=1e-6, pos_target=-1e-8, pct_log=0.0001, use_cuda=False,
                      remesh=False, u_tol=1e-9, remesh_factor=1.01, log_all=False, adjust_starting_point=False,
                      b1=None, b0=None, c1=None, c0=None):
    '''
    integrate the interaction of the probe (connected to a rigid cantilever) with a sample defined by a viscoelastic
    ODE of maximum order N=1, using RK4 time integration
    :param Gg: float instantaneous shear modulus
    :param Ge: float equilibrium shear modulus
    :param Tau: float relaxation time
    :param v: float poisson's ratio
    :param v0: float prescribed probe velocity
    :param h0: float initial position of the probe relative to the sample
    :param R: float radius of the spherical probe
    :param p_func: function describing the point-probe pressure distribution p_func(h, *args)
    :param args: optional arguments for p_func pressure distribution
    :param nr: int number of spatial discretization points
    :param dr: float spatial discretization
    :param dt: float temporal discretization
    :param nt: int number of time discretization points
    :param force_target: float maximum force in simulation
    :param pos_target: float target position of probe
    :param pct_log: float percentage of sim steps to log sim state to console
    :param use_cuda: bool whether to use CUDA cores for calculation acceleration (nearly 1 order of magnitude
    compared against numpy.linalg.solve)
    :param remesh: bool whether to remesh the domain
    :param u_tol: tolerance for the 'infinite' displacement value
    :param remesh_factor: float how large the domain will expand if remeshing
    :param log_all: bool whether to log surface distribution data, default is False
    :param adjust_starting_point: bool whether to adjust starting point
    :param b1: float (optional default is None) first order generalized viscoelastic ODE strain coefficient
    :param b0: float (optional default is None) zeroth order generalized viscoelastic ODE strain coefficient
    :param c1: float (optional default is None) first order generalized viscoelastic ODE stress coefficient
    :param c0: float (optional default is None) zeroth order generalized viscoelastic ODE stress coefficient
    :return: data dict containing sim dataframe and sim parameters
    '''
    saved_args = locals()  # save all function arguments for later

    # discretize domain
    r = np.linspace(1, nr, nr) * dr
    # calculate integrating kernels
    k_ij = np.array([K(r_, r, dr) for r_ in r])
    I = np.eye(r.size)  # make identity matrix
    u = np.zeros(r.shape)  # initialize deformation
    state = np.array([u, h0], dtype='object')  # make state vector

    if use_cuda and torch.cuda.is_available():
        print('Solving on GPU:')
        print(torch.cuda.get_device_name(0))
        torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
        r = torch.linspace(1, nr, nr, device='cuda') * dr
        # TODO: there is a lot of computational overhead in making the cuda version of k_ij, should be fixed easily
        k = torch.zeros([nr, nr], device='cuda')
        for i, row in enumerate(k_ij):
            for j, K_IJ in enumerate(row):
                k[i, j] += K_IJ
        k_ij = k
        I = torch.eye(nr, device='cuda')
        u = torch.zeros([nr, 1], device='cuda')
        h0 = torch.ones([nr, 1], device='cuda') * h0
        v0 = torch.ones([nr, 1], device='cuda') * v0
        state = torch.cat((u, h0), 1)
    else:
        use_cuda = False
        print('GPU unavailable, using CPU')

    if all([_ is None for _ in (b1, b0, c1, c0)]):
        # get generalized viscoelastic coefficients if not explicitly given
        if Ge == 0 and Tau == 0:
            print('assigning elastic coefs')
            b1, b0, c1, c0 = get_coefs_elastic(Gg, v)
        else:
            print('assigning viscoelastic coefs')
            b1, b0, c1, c0 = get_coefs_sls(Gg, Ge, Tau, v)

    print('b1 | b0 | c1 | c0')
    print(b1, b0, c1, c0)

    print('max. r: {}'.format(max(r)))

    # make loggers
    force = np.zeros(nt)
    separation = np.zeros(nt)
    tip_pos = np.zeros(nt)
    time = np.zeros(nt)
    u_inf = np.zeros(nt)

    if log_all:
        U_log = np.zeros((nt, u.size))
        P_log = np.zeros((nt, u.size))
        r_log = np.zeros((nt, u.size))

    if adjust_starting_point:
        print('adjusting starting point...')
        # try first-step
        f = np.inf
        while f > 1e-10:  # while starting force is too large, increase h0
            if use_cuda:  # need to benchmark this
                state[:, 1] *= 2
                state, (p, h) = rk4(state, dt, rhs_N_1_rigid_cuda, r, dr, R, k_ij, I, v0, b1, b0, c1, c0, p_func, *args)
            else:
                state[1] *= 2
                state, (p, h) = rk4(state, dt, rhs_N_1_rigid, r, dr, R, k_ij, I, v0, b1, b0, c1, c0, p_func, *args)
            f = 2 * np.pi * p @ r * dr  # calculate the force
        print('done!')

    # run simulation
    print(' % |  z_tip  |  z_target |  force  |  force_target  |  u(inf,t)')
    start = time_.time()
    for n in range(nt):
        # update the state according to rk4 integration
        if use_cuda:  # need to benchmark this
            state, (p, h) = rk4(state, dt, rhs_N_1_rigid_cuda, r, dr, R, k_ij, I, v0, b1, b0, c1, c0, p_func, *args)
            u = state[:, 0]#.cpu().numpy()
            h0 = state[:, 1][0]#.cpu().numpy()[0]
        else:
            state, (p, h) = rk4(state, dt, rhs_N_1_rigid, r, dr, R, k_ij, I, v0, b1, b0, c1, c0, p_func, *args)
            u, h0 = state
        force[n] = (2 * np.pi * p @ r * dr)  # calculate the force
        # log
        separation[n] = h[0]
        tip_pos[n] = h0
        time[n] = n * dt
        u_inf[n] = u[-1]
        if log_all:
            U_log[n] = u
            P_log[n] = p
            r_log[n] = r
        if n % (pct_log * nt) == 0:
            print('{} | {:.1f} nm | {:.1f} nm | {:.1f} nN | {:.1f} nN | {:.1f} nm'.format(
                n / nt, tip_pos[n] * 1e9, pos_target * 1e9, force[n] * 1e9,
                force_target * 1e9, u[-1] * 1e9))
        # early exit conditions
        if force[n] > force_target or tip_pos[n] < pos_target:
            print('force and target ', force[n], force_target)
            print('pos and target ', tip_pos[n], pos_target)
            force = force[:n + 1]
            separation = separation[:n + 1]
            tip_pos = tip_pos[:n + 1]
            time = time[:n + 1]
            u_inf = u_inf[:n + 1]
            if log_all:
                U_log = U_log[:n + 1]
                P_log = P_log[:n + 1]
                r_log = r_log[:n + 1]
            break
        if remesh and abs(u[-1]) > u_tol:
            # remake domain
            print(max(r))
            dr *= remesh_factor
            # discretize domain
            r = np.linspace(1, nr, nr) * dr
            print(max(r))
            # calculate integrating kernels
            k_ij = np.array([K(r_, r, dr) for r_ in r])
            I = np.eye(r.size)  # make identity matrix
            u = np.zeros(r.shape)  # initialize deformation
            state = np.array([u, h0], dtype='object')  # make state vector
            print('this is the issue, in the line above.  u needs to be properly defined using the previous values')
            if use_cuda and torch.cuda.is_available():
                torch.set_default_tensor_type(
                    torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
                r = torch.linspace(1, nr, nr, device='cuda') * dr
                k = torch.zeros([nr, nr], device='cuda')
                for i, row in enumerate(k_ij):
                    for j, K_IJ in enumerate(row):
                        k[i, j] += K_IJ
                k_ij = k
                I = torch.eye(nr, device='cuda')
                u = torch.zeros([nr, 1], device='cuda')
                h0 = torch.ones([nr, 1], device='cuda') * h0
                v0 = torch.ones([nr, 1], device='cuda') * v0
                state = torch.cat((u, h0), 1)

    print('done in {:.0f}s'.format(time_.time() - start))
    # save sim data
    inden = np.zeros(time.size)
    contact_index = np.argmin(force)
    inden[contact_index:] = tip_pos[contact_index] - tip_pos[contact_index:]
    force_rep = np.zeros(time.size)
    force_rep[contact_index:] = force[contact_index:] - force[contact_index]
    df = pd.DataFrame({'time': time, 'separation': separation, 'tip_pos': tip_pos,
                       'force': force, 'inden': inden, 'force_rep': force_rep,
                       'deformation': separation - tip_pos, 'u_inf': u_inf})
    data = {'df': df, 'sim_arguments': saved_args, 'log_all': 0}
    if log_all:
        data['log_all'] = {'u': U_log, 'p': P_log, 'r': r_log}
    return data




def simulate_prescribed_N1(Gg, Ge, Tau, v, v_t, h0, R, p_func, *args,
                           nr=int(1e3), dr=1.5e-9, dt=1e-4, pct_log=0.01):
    '''
    integrate the interaction of the probe (connected to a rigid cantilever) with a sample defined by a viscoelastic
    ODE of maximum order N=1, using RK4 time integration with the probe position prescribed as a function of time
    :param Gg: float instantaneous shear modulus
    :param Ge: float equilibrium shear modulus
    :param Tau: float relaxation time
    :param v: float poisson's ratio
    :param v_t: np.array prescribed probe velocity as a function of time
    :param h0: float initial position of the probe relative to the sample
    :param R: float radius of the spherical probe
    :param p_func: function describing the point-probe pressure distribution p_func(h, *args)
    :param args: optional arguments for p_func pressure distribution
    :param nr: int number of spatial discretization points
    :param dr: float spatial discretization
    :param dt: float temporal discretization
    :param pct_log: float percentage of sim steps to log sim state to console
    :return: data dict containing sim dataframe and sim parameters
    '''
    saved_args = locals()  # save all function arguments for later

    # discretize domain
    r = np.linspace(1, nr, nr) * dr
    # calculate integrating kernels
    k_ij = np.array([K(r_, r, dr) for r_ in r])
    I = np.eye(r.size)  # make identity matrix
    u = np.zeros(r.shape)  # initialize deformation
    state = np.array([u, h0], dtype='object')  # make state vector

    # get generalized viscoelastic coefficients
    if Ge == 0 and Tau == 0:
        b1, b0, c1, c0 = get_coefs_elastic(Gg, v)
    else:
        b1, b0, c1, c0 = get_coefs_sls(Gg, Ge, Tau, v)

    print('b1 | b0 | c1 | c0')
    print(b1, b0, c1, c0)
    print('max. r: {}'.format(max(r)))

    # make loggers
    nt = v_t.size
    force = np.zeros(nt)
    separation = np.zeros(nt)
    tip_pos = np.zeros(nt)
    time = np.zeros(nt)
    u_inf = np.zeros(nt)

    # run simulation
    print(' % |  z_tip  |  force  |  u(inf,t) | t')
    start = time_.time()
    for n in range(nt):
        # update the state according to rk4 integration
        v0 = v_t[n]
        state, (p, h) = rk4(state, dt, rhs_N_1_rigid, r, dr, R, k_ij, I, v0, b1, b0, c1, c0, p_func, *args)
        u, h0 = state
        force[n] = (2 * np.pi * p @ r * dr)  # calculate the force
        # log
        separation[n] = h[0]
        tip_pos[n] = h0
        time[n] = n * dt
        u_inf[n] = u[-1]
        if n % (pct_log * nt) == 0:
            print('{} | {:.1f} nm | {:.1f} nN | {:.1f} nm | {:.1f} s'.format(
                n / nt, tip_pos[n] * 1e9, force[n] * 1e9, u[-1] * 1e9, n * dt))

    print('done in {:.0f}s'.format(time_.time() - start))
    # save sim data
    inden = np.zeros(time.size)
    contact_index = np.argmin(force)
    inden[contact_index:] = tip_pos[contact_index] - tip_pos[contact_index:]
    force_rep = np.zeros(time.size)
    force_rep[contact_index:] = force[contact_index:] - force[contact_index]
    df = pd.DataFrame({'time': time, 'separation': separation, 'tip_pos': tip_pos,
                       'force': force, 'inden': inden, 'force_rep': force_rep,
                       'deformation': separation - tip_pos, 'u_inf': u_inf})
    data = {'df': df, 'sim_arguments': saved_args, 'log_all': 0}
    return data



def rk4_mixed(state_vectors, state_scalars, dt, rhs_func_mixed, *args):
    '''
    Runge Kutta 4th order time integration update scheme
    :param state_vectors: (vector) vector variables to integrate
    :param state_scalars: (vector like) vector of scalar variables to integrate
    :param dt: (float) time discretization
    :param rhs_func_mixed: function to calculate the time derivatives of the state of form
    rhs_func(state_vectors, state_scalars, *args) returns (time derivative of state vector, (extra stuff))
    :param args: (optional) arguments for the rhs_func
    :return: (numpy array) integrated state at t=t+dt, (extra stuff calculated at final step)
    '''
    k1v, k1s, _, _ = rhs_func_mixed(state_vectors, state_scalars, *args)
    k2v, k2s, _, _ = rhs_func_mixed(state_vectors + dt * k1v / 2, state_scalars + dt * k1s,
                                    *args)
    k3v, k3s, _, _ = rhs_func_mixed(state_vectors + dt * k2v / 2, state_scalars + dt * k2s / 2,
                                    *args)
    k4v, k4s, extra_vectors, extra_scalars = rhs_func_mixed(state_vectors + dt * k3v,
                                                            state_scalars + dt * k3s, *args)
    return (state_vectors + (k1v + 2 * k2v + 2 * k3v + k4v) * dt / 6,
            state_scalars + (k1s + 2 * k2s + 2 * k3s + k4s) * dt / 6,
            extra_vectors, extra_scalars)

def rhs_mixed(state_vectors, state_scalars, t, b1, b0, c1, c0, r, dr, R, k_ij, I, use_cuda, f_exc_func, vb_func,
              m, k_eff, Q0, w0, p_func, *args):
    '''
    calculates the derivative of the state_vectors and state_scalars (surface deformation and probe position)
    assuming that the probe position is governed by a simple harmonic oscillator with a prescribed base velocity
    and base excitation - also assuming that the viscoelastic ODE is order N=1
    :param state_vectors: array state vector (deformation)
    :param state_scalars: array state scalars (zbase, ztip, vtip)
    :param t: float time
    :param b1: float viscoelastic coefficient b1
    :param b0: float viscoelastic coefficient b0
    :param c1: float viscoelastic coefficient c1
    :param c0: float viscoelastic coefficient c0
    :param r: float radial spatial axis
    :param dr: float spatial discretization
    :param R: float spherical probe radius
    :param k_ij: matrix (nr x nr) integrating kernel
    :param I: numpy matrix (nr x nr) identity matrix
    :param use_cuda: bool True to use cuda
    :param f_exc_func: function to calculate the excitation force on the cantilever, uses args: (t, zt, zb, f_ts, k_eff)
    :param vb_func: function to calculate the velocity of the cantilever base, uses args: (t, zt, zb, f_ts, k_eff)
    :param m: effective mass of first cantilever mode
    :param k_eff: effective spring constant of first cantilever mode
    :param Q0: Q factor of first cantilever mode
    :param w0: resonant frequency of first cantilever mode
    :param p_func: function for calculating sphere-point pressure distribution p_func(h, *args)
    :param args: optional arguments for p_func
    :return: numpy array derivative of state vector (deformation rate, probe velocity)
    '''
    u = state_vectors
    zb, zt, vt = state_scalars
    # calculate state derivatives
    # first calculate the derivatives for the surface
    h = h_calc(zt, u, f_sphere, r, R)  # calculate the separation
    p, p_h = p_func(h, *args)  # calculate the pressure and derivative
    f_ts = 2 * np.pi * p @ r * dr  # calculate the tip-sample force
    # calculate LHS G matrix
    G_ij = c1 * dr * k_ij * p_h * r - b1 * I
    d_j = c1 * vt * dr * k_ij @ (p_h * r) + c0 * dr * k_ij @ (p * r) + b0 * u
    # solve the system of equations to calculate u_dot
    if use_cuda:
        u_dot = torch.linalg.solve(G_ij, d_j)
    else:
        u_dot = np.linalg.solve(G_ij, d_j)
    # then calculate the derivatives of the cantilever
    f_exc = f_exc_func(t, zt, zb, f_ts, k_eff)
    zb_dot = vb_func(t, zt, zb, f_ts, k_eff)
    zt_dot = vt
    vt_dot = 1 / m * (f_exc + f_ts + k_eff * (zb - zt) - m * Q0 / w0 * vt)
    # returns: ((state vectors, state scalars), (extra vectors, extra scalars) (won't be used for integration))
    if use_cuda:
        return (u_dot, torch.tensor([zb_dot, zt_dot, vt_dot]), (p, h), torch.tensor([f_ts, f_exc, vt_dot]))
    else:
        return (u_dot, np.array([zb_dot, zt_dot, vt_dot]), (p, h), np.array([f_ts, f_exc, vt_dot]))

def simulate_cantilever_N1(Gg, Ge, Tau, v, zt, vt, zb, R, f_exc_func, vb_func, p_func, *args,
                           nr=int(1e3), dr=1.5e-9, dt_factor=1e-10, nt=int(1e6), w0=10e3, Q0=100, k_eff=1,
                           force_target=np.inf, pos_target=-1e-8, pct_log=0.0001, use_cuda=False, log_all=False):
    '''
    integrate the interaction of the probe (connected to a SHO cantilever) with a sample defined by a viscoelastic
    ODE of maximum order N=1, using RK4 time integration
    :param Gg: float instantaneous shear modulus
    :param Ge: float equilibrium shear modulus
    :param Tau: float relaxation time
    :param v: float poisson's ratio
    :param zt: float initial tip position
    :param vt: float initial probe velocity
    :param zb: float initial probe position
    :param R: float radius of the spherical probe
    :param f_exc_func: function to calculate the excitation force on the cantilever, uses args: (t, zt, zb, f_ts, k_eff)
    :param vb_func: function to calculate the velocity of the cantilever base, uses args: (t, zt, zb, f_ts, k_eff)
    :param p_func: function describing the point-probe pressure distribution p_func(h, *args)
    :param args: optional arguments for p_func pressure distribution
    :param nr: int number of spatial discretization points
    :param dr: float spatial discretization
    :param dt: float temporal discretization
    :param nt: int number of time discretization points
    :param force_target: float maximum force in simulation
    :param pos_target: float target position of probe
    :param pct_log: float percentage of sim steps to log sim state to console
    :param use_cuda: bool whether to use CUDA cores for calculation acceleration (nearly 1 order of magnitude
    compared against numpy.linalg.solve)
    :param log_all: bool whether to log surface distribution data, default is False
    :return: data dict containing sim dataframe and sim parameters
    '''
    dt = abs(dt_factor / vt)
    saved_args = locals()  # save all function arguments for later
    # discretize domain
    r = np.linspace(1, nr, nr) * dr
    # calculate integrating kernels
    k_ij = np.array([K(r_, r, dr) for r_ in r])
    I = np.eye(r.size)  # make identity matrix
    u = np.zeros(r.shape)  # initialize deformation

    # get generalized viscoelastic coefficients
    if Ge == 0 and Tau == 0:
        b1, b0, c1, c0 = get_coefs_elastic(Gg, v)
    else:
        b1, b0, c1, c0 = get_coefs_sls(Gg, Ge, Tau, v)

    print('b1 | b0 | c1 | c0')
    print(b1, b0, c1, c0)
    print('max. r: {}'.format(max(r)))

    # forces
    f_ts = 0
    f_exc = 0

    # probe variables
    m = k_eff / w0 ** 2
    vb = vb_func(0, zt, zb, f_ts, k_eff)

    # form state
    state_vectors = u
    state_scalars = np.array([zb, zt, vt])

    # make loggers
    central_deformation = np.zeros(nt)
    zb_zt_vt = np.zeros((nt, state_scalars.size))
    f_ts_f_exc_at = np.zeros((nt, 3))
    time = np.zeros(nt)
    u_inf = np.zeros(nt)

    log_all = False
    if log_all:
        U_log = np.zeros((nt, u.size))
        P_log = np.zeros((nt, u.size))
        H_log = np.zeros((nt, u.size))
        r_log = np.zeros((nt, u.size))

    if use_cuda and torch.cuda.is_available():
        print('Solving on GPU:')
        print(torch.cuda.get_device_name(0))
        torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
        r = torch.linspace(1, nr, nr, device='cuda') * dr
        # TODO: there is a lot of computational overhead in making the cuda version of k_ij, should be fixed easily
        k = torch.zeros([nr, nr], device='cuda')
        for i, row in enumerate(k_ij):
            for j, K_IJ in enumerate(row):
                k[i, j] += K_IJ
        k_ij = k
        I = torch.eye(nr, device='cuda')
        u = torch.zeros(nr, device='cuda')

        # forces
        f_ts = torch.tensor(f_ts, device='cuda')
        f_exc = torch.tensor(f_exc, device='cuda')
        # probe variables
        zt = torch.tensor(zt, device='cuda')
        zb = torch.tensor(zb, device='cuda')
        vb = torch.tensor(vb, device='cuda')
        vt = torch.tensor(vt, device='cuda')

        # form state
        state_vectors = u
        state_scalars = torch.tensor([zb, zt, vt], device='cuda')

        # make loggers
        central_deformation = torch.zeros(nt, device='cuda')
        zb_zt_vt = torch.zeros((nt, state_scalars.size()[0]), device='cuda')
        f_ts_f_exc_at = torch.zeros((nt, state_scalars.size()[0]), device='cuda')
        time = torch.zeros(nt, device='cuda')
        u_inf = torch.zeros(nt, device='cuda')

        # cantilever, probe, and sample variables
        R = torch.tensor(R, device='cuda')
        b1, b0 = torch.tensor(b1, device='cuda'), torch.tensor(b0, device='cuda')
        c1, c0 = torch.tensor(c1, device='cuda'), torch.tensor(c0, device='cuda')
        w0 = torch.tensor(w0, device='cuda')
        Q0 = torch.tensor(Q0, device='cuda')
        k_eff = torch.tensor(k_eff, device='cuda')
        m = torch.tensor(m, device='cuda')

        if log_all:
            U_log = torch.zeros((nt, u.size()[0]), device='cuda')
            P_log = torch.zeros((nt, u.size()[0]), device='cuda')
            H_log = torch.zeros((nt, u.size()[0]), device='cuda')
            r_log = torch.zeros((nt, u.size()[0]), device='cuda')

    # solve
    print(' % |  z_tip  |  z_target |  force  |  force_target  |  u(inf,t)')
    start = time_.time()
    for n in range(nt):
        t = n * dt  # advance time
        state_vectors, state_scalars, extra_vectors, extra_scalars = rk4_mixed(state_vectors, state_scalars, dt,
                                                                               rhs_mixed, t, b1, b0, c1, c0, r, dr,
                                                                               R, k_ij, I, use_cuda, f_exc_func,
                                                                               vb_func, m, k_eff, Q0, w0,
                                                                               p_func, *args)
        # log
        zb_zt_vt[n] = state_scalars
        f_ts_f_exc_at[n] = extra_scalars
        central_deformation[n] = state_vectors[0]
        time[n] = t
        u_inf[n] = state_vectors[-1]

        if n % (pct_log * nt) == 0:
            print('{} | {:.1f} nm | {:.1f} nN | {:.1f} nm | {:.1f} s'.format(
                n / nt, zb_zt_vt[n, 0] * 1e9, f_ts_f_exc_at[n, 0] * 1e9, state_vectors[-1] * 1e9, n * dt))
        if log_all:
            U_log[n] = state_vectors[0]
            P_log[n] = extra_vectors[0]
            H_log[n] = extra_vectors[1]
            r_log[n] = r
        if zb_zt_vt[n, 0] < pos_target or f_ts_f_exc_at[n, 0] > force_target:
            zb_zt_vt = zb_zt_vt[:n + 1]
            f_ts_f_exc_at = f_ts_f_exc_at[:n + 1]
            central_deformation = central_deformation[:n + 1]
            time = time[:n + 1]
            u_inf = u_inf[:n + 1]
            if log_all:
                U_log = U_log[:n + 1]
                P_log = P_log[:n + 1]
                H_log = H_log[:n + 1]
                r_log = r_log[:n + 1]
            break
    if use_cuda:
        zb_zt_vt = zb_zt_vt.cpu().numpy()
        f_ts_f_exc_at = f_ts_f_exc_at.cpu().numpy()
        central_deformation = central_deformation.cpu().numpy()
        time = time.cpu().numpy()
        k_eff = k_eff.cpu().numpy()
        u_inf = u_inf.cpu().numpy()
        if log_all:
            U_log = U_log.cpu().numpy()
            P_log = P_log.cpu().numpy()
            H_log = H_log.cpu().numpy()
            r_log = r_log.cpu().numpy()
    zb, zt, vt = zb_zt_vt.T
    f_ts, f_exc, at = f_ts_f_exc_at.T
    defl = zt - zb
    inden = - (zb + defl)  # this must be reversed as compared to the true relationship as zbase increases in AFM

    print('done in {:.0f}s'.format(time_.time() - start))
    # save sim data
    separation = zt - central_deformation
    df = pd.DataFrame({'time': time, 'separation_true': separation, 'tip_pos': zt, 'tip_vel': vt, 'tip_accel': at,
                       'base_pos': zb, 'force_ts_true': f_ts, 'force_ts_afm': defl * k_eff, 'defl': defl,
                       'inden_afm': inden, 'sample_pos': central_deformation, 'u_inf': u_inf})
    data = {'df': df, 'sim_args': saved_args, 'log_all': 0}
    if log_all:
        data['log_all'] = {'U_log': U_log, 'P_log': P_log, 'H_log': H_log, 'r_log': r_log}
    return data


def calculate_fluid_pressure(h_dot, h, r, dr, eta, J):
    """calculates the fluid pressure distribution by integration of the reynolds equation assuming the gap between the probe
    and the sample is small relative to the probe radius

    Args:
        h_dot (numpy array): time derivative of the gap
        h (numpy array): gap
        r (numpy array): radial axis
        dr (float): radial discretization step
        eta (float): fluid viscosity
        J (numpy matrix): matrix of ones with zeros above the diagonal for integration

    Returns:
        numpy array: radial fluid pressure distribution
    """
    integral1 = J * dr * 12 * eta @ (r * h_dot)
    integral1 -= integral1[0]  # apply symmetry boundary conditions (i.e. radial derivative of pressure distribution is zero at r=0)
    integral2 = J * dr @ (integral1 / (r * h ** 3))  # calculate the fluid pressure
    return integral2 - integral2[-1]  # apply boundary conditions where the pressure is zero at r=infinity


def solve_solid(b1, b0, c1, c0, k_ij, I, r, dr, u, v_tip, p, p_h):
    """solve the solid mechanics problem
    Args:
        b1 (float): first strain coefficient of the linear viscoelastic ode
        b0 (float): 0th strain coefficient of the linear viscoelastic ode
        c1 (float): first stress coefficient of the linear viscoelastic ode
        c0 (float): 0th stress coefficient of the linear viscoelastic ode
        k_ij (numpy matrix): integrating kernel, part of green's function of linear elastic half-space
        I (numpy matrix): dirac delta function integrating kernel
        r (numpy array): radial axis
        dr (float): radial discretization step
        u (numpy array): deformation profile
        v_tip (float): velocity of the tip
        p (numpy array): pressure distribution profile
        p_h (numpy array): derivative of the pressure distribution profile with respect to the gap
    Returns:
        numpy array: derivatives of the deformation profile and gap profile
    """
    G_ij = c1 * dr * k_ij * p_h * r - b1 * I  # lhs matrix of the integral equation
    d_j = c1 * v_tip * dr * (p_h * r) @ k_ij + c0 * dr * (p * r) @ k_ij + b0 * u  # lhs vector of the integral equation
    u_dot = np.linalg.solve(G_ij, d_j)  # directly obtain the derivative of the deformation by inverting the integral equation
    h_dot = v_tip - u_dot  # calculate the derivative of the gap assuming the tip is rigid
    return u_dot, h_dot

def solve_fluid(eta, b1, b0, c1, c0, k_ij, I, J, R, r, dr, u, h0, v_tip, p_fluid_init, h_dot_init, dt, max_iter, err_tol, p_func, *args):
    """solve the fluid mechanics problem

    Args:
        b1 (float): first strain coefficient of the linear viscoelastic ode
        b0 (float): 0th strain coefficient of the linear viscoelastic ode
        c1 (float): first stress coefficient of the linear viscoelastic ode
        c0 (float): 0th stress coefficient of the linear viscoelastic ode
        k_ij (numpy matrix): integrating kernel, part of green's function of linear elastic half-space
        I (numpy matrix): dirac delta function integrating kernel
        R (float): probe radius
        r (numpy array): radial axis
        dr (float): radial discretization step
        u (numpy array): deformation profile
        h0 (float): position of the tip relative to the equilibrium position of the sample
        v_tip (float): velocity of the tip
        p_fluid_init (numpy array): initial guess for the fluid pressure
        h_dot_init (numpy array): initial guess for the gap velocity
        dt (float): time step
        max_iter (int): maximum number of iterations for convergence
        err_tol (float): maximum error tolerance for convergence
        p_func (function): pressure function taking optional arguments *args

    Returns:
        tuple of numpy arrays: deformation, deformation rate, gap, gap rate, fluid pressure, total pressure, number of iterations, convergence error
    """
    p_fluid = p_fluid_init.copy()  # initial guess for the fluid pressure
    h_dot = h_dot_init.copy()  # initial guess for the gap velocity
    h_init = h0 + r ** 2 / (2 * R) - u  # calculate the gap profile
    h = h_init.copy()  # initial guess for the gap profile
    
    err = np.inf
    num_iter = 0
    while num_iter < max_iter and err > err_tol:  # iterate until error is satisfied or out of bounds
        num_iter += 1
        p_fluid_h = p_fluid / (h_dot * dt)  # calculate the derivative of the fluid pressure with respect to the gap
        p_surface, p_surface_h = p_func(h, *args)  # calculate the surface pressure and its gap derivative
        p, p_h = p_fluid + p_surface, p_fluid_h + p_surface_h  # calculate the total  pressure and its gap derivative
        u_dot, h_dot = solve_solid(b1, b0, c1, c0, k_ij, I, r, dr, u, v_tip, p, p_h)  # solve the solid mechanics problem
        
        # calculate the new gap and deformation profiles
        h = h_init + h_dot * dt
        u += u_dot * dt
        
        # calculate the fluid pressure profile due to the new gap profile
        p_fluid = calculate_fluid_pressure(h_dot, h, r, dr, eta, J)
        
        # calculate the error
        err = np.sum(abs(p_fluid - p_fluid_init))
        
        # propogate new values to initial values for next iteration
        p_fluid_init = p_fluid.copy()
        h_dot_init = h_dot.copy()
    return u, u_dot, h, h_dot, p_fluid, p, num_iter, err
        
def rk4_fluid(eta, b1, b0, c1, c0, k_ij, I, J, R, r, dr, u, h0, v_tip, p_fluid_init, h_dot_init, dt, max_iter, err_tol, p_func, *args):
    """update the surface deformation, gap profile, and probe position using the 4th order Runge-Kutta method

    Args:
        b1 (float): first strain coefficient of the linear viscoelastic ode
        b0 (float): 0th strain coefficient of the linear viscoelastic ode
        c1 (float): first stress coefficient of the linear viscoelastic ode
        c0 (float): 0th stress coefficient of the linear viscoelastic ode
        k_ij (numpy matrix): integrating kernel, part of green's function of linear elastic half-space
        I (numpy matrix): dirac delta function integrating kernel
        R (float): probe radius
        r (numpy array): radial axis
        dr (float): radial discretization step
        u (numpy array): deformation profile
        h0 (float): position of the tip relative to the equilibrium position of the sample
        v_tip (float): velocity of the tip
        p_fluid_init (numpy array): initial guess for the fluid pressure
        h_dot_init (numpy array): initial guess for the gap velocity
        dt (float): time step
        max_iter (int): maximum number of iterations for convergence
        err_tol (float): maximum error tolerance for convergence
        p_func (function): function that returns the surface pressure and its gap derivative with optional arguments *args

    Returns:
        tuple of numpy arrays: new deformation, new gap, new tip position, new fluid pressure, new total pressure
    """
    h = h0 + r ** 2 / (2 * R) - u
    k1 = solve_fluid(eta, b1, b0, c1, c0, k_ij, I, J, R, r, dr, u, h0, v_tip, p_fluid_init, h_dot_init, dt, max_iter, err_tol, p_func, *args)
    k2 = solve_fluid(eta, b1, b0, c1, c0, k_ij, I, J, R, r, dr, u + k1[1] * dt / 2, h0 + v_tip * dt / 2, v_tip, k1[4], k1[3], dt, max_iter, err_tol, p_func, *args)
    k3 = solve_fluid(eta, b1, b0, c1, c0, k_ij, I, J, R, r, dr, u + k2[1] * dt / 2, h0 + v_tip * dt / 2, v_tip, k2[4], k2[3], dt, max_iter, err_tol, p_func, *args)
    k4 = solve_fluid(eta, b1, b0, c1, c0, k_ij, I, J, R, r, dr, u + k3[1] * dt, h0 + v_tip * dt, v_tip, k3[4], k3[3], dt, max_iter, err_tol, p_func, *args)
    return (u + (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) * dt / 6, 
            h + (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) * dt / 6,
            h0 + v_tip * dt, k4[4], k4[5], k4[-2])
    

def p_brenner(h, v_tip, eta, R):
    """calculate the surface pressure and its gap derivative using the Brenner model for rigid surfaces

    Args:
        h (numpy array): gap profile
        v_tip (float): tip velocity
        eta (float): fluid viscosity
        R (float): probe radius

    Returns:
        tuple of numpy arrays: pressure profile and its gap derivative
    """
    factor = 3 * eta * v_tip * R
    return (-factor / h ** 2, 2 * factor / h ** 3)

def calc_force(p_total, r, dr):
    """calculate the force on the surface, which is the same as the force on the probe due to the lubrication approximation

    Args:
        p_total (numpy array): pressure profile
        r (numpy array): radial axis
        dr (float): radial discretization step

    Returns:
        float: force on the surface
    """
    return 2 * np.pi * dr * p_total @ r

    
def simulate_fluid(Gg, Ge, Tau, v, eta, v_tip, h0, R, p_func, *args, nr=int(1e3), dr_factor=1.1, dt_factor=1e-10, nt=int(1e6), pos_target=-1e-8, force_target_mult=1, pct_log=0.0001):
    
    saved_args = locals()  # save all function arguments for later
    
    # get generalized viscoelastic coefficients and calculate the target force using the Hertzian model
    if Ge == 0 and Tau == 0:
        b1, b0, c1, c0 = get_coefs_elastic(Gg, v)
        force_target = force_target_mult * hertzian(abs(pos_target), Gg, v, R)
        
    else:
        b1, b0, c1, c0 = get_coefs_sls(Gg, Ge, Tau, v)
        force_target = force_target_mult * hertzian(abs(pos_target), Ge, v, R)
        
    dr = dr_factor * R / nr  # radial discretization step
    dt = abs(dt_factor / v_tip)
    
    r = np.linspace(1, nr, nr) * dr  # radial axis
    k_ij = np.array([K(r_, r, dr) for r_ in r])  # integrating kernel, part of green's function of linear elastic half-space
    I = np.eye(r.size)  # dirac delta function integrating kernel
    J = np.tril(np.ones(I.shape))  # lower triangular matrix for integration
    
    u = np.zeros(r.shape)
    
    # initially solve the hydrodynamic pressure for the case of no deformation
    h = h0 + r ** 2 / (2 * R) - u
    p_fluid_init = p_brenner(h, v_tip, eta, R)[0]
    h_dot_init = v_tip * np.ones(r.shape)
    
    # make loggers
    force = np.zeros(nt)
    force_fluid = np.zeros(nt)
    separation = np.zeros(nt)
    deformation = np.zeros(nt)
    position = np.zeros(nt)
    time = np.zeros(nt)
    U_log = np.zeros((nt, u.size))
    P_log = np.zeros((nt, u.size))
    P_fluid_log = np.zeros((nt, u.size))
    
    for i in range(nt):
        # update the fluid pressure and gap velocity using the 4th order Runge-Kutta method
        h_init = h0 + r ** 2 / (2 * R) - u
        u, h, h0, p_fluid, p, iters = rk4_fluid(eta, b1, b0, c1, c0, k_ij, I, J, R, r, dr, u, h0, v_tip, p_fluid_init, h_dot_init, dt, 10, 1e-10, p_func, *args)

        
        # propogate new values to initial values for next iteration
        p_fluid_init = p_fluid
        h_dot_init = (h - h_init) / dt
        
        # log
        force[i] = calc_force(p, r, dr)
        force_fluid[i] = calc_force(p_fluid, r, dr)
        separation[i] = h0 - u[0]
        position[i] = h0
        deformation[i] = u[0]
        time[i] = i * dt
        U_log[i] = u
        P_log[i] = p
        P_fluid_log[i] = p_fluid
        
        # log the progress
        if i % int(nt * pct_log) == 0:
            print('step: {}, num iters: {}, pos: {:.2f} nm, force: {:.2f} nN'.format(i, iters, position[i] * 1e9, force[i] * 1e9))
            
        # early exit conditions
        if force[i] > force_target or position[i] < pos_target:
            print('force: {:.2f} nN, target: {:.2f} nN'.format(force[i] * 1e9, force_target * 1e9))
            print('pos: {:.2f} nm, target: {:.2f} nm'.format(position[i] * 1e9, pos_target * 1e9))
            force = force[:i + 1]
            separation = separation[:i + 1]
            position = position[:i + 1]
            time = time[:i + 1]
            deformation = deformation[:i + 1]
            force_fluid = force_fluid[:i + 1]
            U_log = U_log[:i + 1]
            P_log = P_log[:i + 1]
            break
    
    df = pd.DataFrame({'time': time, 'separation': separation, 'position': position, 'force': force, 'deformation': deformation, 'force_fluid': force_fluid})
    data = {'df': df, 'sim_arguments': saved_args, 'log_all': {'u': U_log, 'p': P_log, 'r': r, 'p_fluid': P_fluid_log}}
    return data

class ContactSim:
    """parent class for the contact simulation for a rigid sphere on a flat, linear elastic or linear viscoelastic semi-infinite half-space
    with primary assumption that the initial separation and maximum deformation are smaller than the probe radius and that the probe velocity is fixed
    further, the surface starts at its equilibrium position (i.e. is undeformed)
    """
    
    def __init__(self, Gg, Ge, Tau, poissons, R, v_tip, h0, h0_target, p_func, *args, nr=int(1e3), dr_factor=1.1, dt_factor=1e-10, nt=int(1e6), log_all=False, pct_log=0.1):
        """initialize the simulation class with the parameters for the simulation

        Args:
            Gg (float): instantaneous (glassy) shear modulus of the material
            Ge (float): equilibrium (rubbery) shear modulus of the material
            Tau (float): relaxation time of the material
            poissons (float): poissons ratio of the material
            R (float): probe radius
            v_tip (float): velocity of the probe
            h0 (float): initial separation between the probe and the equilibrium position of the surface
            h0_target (float): target position of the probe relative to the equilibrium position of the surface
            p_func (_type_): function that describes the pressure on the surface as a function of the gap height and optional arguments, *args, must
            return the pressure and the derivative of the pressure with respect to the gap height
            nr (int, optional): number of elements in the radial domain. Defaults to int(1e3).
            dr_factor (float, optional): multiplying this to the probe radius gives the largest radial distance in the domain. Defaults to 1.1.
            dt_factor (float, optional): the distance that the probe is advanced per time step. Defaults to 1e-10.
            nt (int, optional): maximum number of timesteps in the simulation. Defaults to int(1e6).
            log_all (bool, optional): whether or not to log the field variables. Defaults to False.
            pct_log (float, optional): frequency of console logging as a percent of the total estimated number of steps. Defaults to 0.1.

        Raises:
            ValueError: if the target probe position is less than -R/10, the simulation will not be valid
        """
        # general solid parameters for the generalized linear viscoelastic ode
        self.b1 = 0
        self.b0 = 0
        self.c1 = 0
        self.c0 = 0
        
        # logging field equations, usually not on
        self.log_all = log_all
        self.pct_log = pct_log
        
        # get probe parameters
        self.R = R
        self.v_tip = v_tip
        self.h0 = h0
        self.h0_init = h0
        if h0_target < - self.R / 10:
            raise ValueError('h0_target must be greater than -R/10')
        self.h0_target = h0_target
        
        # get pressure function parameters
        self.p_func = p_func
        self.args = args
        
        # define simulation variables
        self.dr = dr_factor * self.R / nr  # INCREASE THIS TO INCREASE THE ACCURACY OF THE FLUID SOLUTION
        self.dt = abs(dt_factor / self.v_tip)  # decrease this to reduce the error in the time step
        self.nt = nt
        self.r = np.linspace(self.dr, nr * self.dr, nr)
        self.k_ij = np.array([K(r_, self.r, self.dr) for r_ in self.r])
        
        # making the solution variable
        self.solution = {'data': None, 'Gg': Gg, 'Ge': Ge, 'Tau': Tau, 'poissons': poissons, 'R': R, 'v_tip': v_tip, 'pressure_func': p_func, 
                         'pressure_func_args': args, 'field_variables': None}
        
    def make_loggers(self):
        """make the loggers for the simulation
        """
        self.force = np.zeros(self.nt)
        self.separation = np.zeros(self.nt)
        self.position = np.zeros(self.nt)
        self.deformation = np.zeros(self.nt)
        self.time = np.zeros(self.nt)
        
    def make_field_loggers(self):
        """make the loggers for the field variables
        """
        self.U = np.zeros((self.nt, self.r.size))
        self.H = np.zeros((self.nt, self.r.size))
        self.P = np.zeros((self.nt, self.r.size))
        
    def calculate_gap(self, u, h0):
        """calculate the gap height as a function of the deformation and the separataion between the spherical probe and the equilibrium position of the surface

        Args:
            u (numpy array): deformation
            h0 (numpy array): separataion between the probe and the equilibrium position of the surface

        Returns:
            numpy array: gap height
        """
        return h0 + self.r ** 2 / (2 * self.R) - u
    
    def calculate_force(self, pressure):
        """calculate the force on the surface as a function of the pressure, if the pressure on the surface does not vary with the gap height,
        this calculation gives the force on the probe as well.  for most surface forces, this is fine - this includes the fluid lubrication force
        using the reynolds equation

        Args:
            pressure (numpy array): pressure profile on the surface

        Returns:
            float: force (positive if repulsive, negative if attractive)
        """
        return 2 * np.pi * self.r @ pressure * self.dr
    
    def console_log(self, step):
        """log the simulation to the console at the specified step

        Args:
            step (int): current simulation step
        """
        if step % int((self.h0_init - self.h0_target) / (self.v_tip * self.dt) * self.pct_log) == 0:
            print('step: {} | h0: {:.1f} (nm) | h0_target: {:.1f} (nm) | force: {:.1f} (nN) | separation: {:.1f} (nm) | time: {:.3f} (s)'.format(step, self.h0 * 1e9, self.h0_target * 1e9, self.force[step] * 1e9, self.separation[step] * 1e9, self.time[step]))
        
    def solve(self):
        raise NotImplementedError
    
class ContactSimIter(ContactSim):
    
    def __init__(self, Gg, Ge, Tau, poissons, eta, R, v_tip, h0, h0_target, p_func, *args, nr=int(1e3), dr_factor=1.1, dt_factor=1e-10, nt=int(1e6), log_all=False, pct_log=0.1):
        """use explicit time stepping to ITERATIVELY solve the COUPLED PDEs for the interaction of a rigid, spherical probe with a NEWTONIAN FLUID 
        lubricating a semi-infinite, linear elastic or linear viscoelastic half-space.  the fluid iteration is inspired by the solution of 
        Davis, Serayssol, and Hinch as well as Wang and Frechette, the solid iteration is reappropraited from our generalized ode solution that is used in the 
        implicit ode solver. The primary assumption is that the initial separation and the maximum deformation are smaller than the probe radius, 
        so that the Derjaguin approximation is valid for the surface forces and the lubrication approximation (Reynolds equation) is valid for the fluid forces.  
        Both of these allow the force on the probe to be calculated as the integral of the pressure on the surface

        Args:
            Gg (float): instantaneous (glassy) shear modulus of the surface
            Ge (float): equilibrium (rubbery) shear modulus of the surface
            Tau (float): relaxation time of the surface
            poissons (float): poissons ratio of the surface
            eta (float): viscosity of the fluid
            R (float): radius of the probe
            v_tip (float): tip velocity
            h0 (float): separation between the probe and the equilibrium position of the surface at r=0 (i.e. global position of the tip of the probe)
            h0_target (float): target position of the tip of the probe
            p_func (_type_): function used to calculate the pressure on the surface with optional args, must be of the form p_func(h, *args) and return a tuple
            of the pressure and the derivative of the pressure with respect to the gap height
            nr (int, optional): number of points in the domain. Defaults to int(1e3).
            dr_factor (float, optional): this multiplied with the probe radius gives the maximum radial distance in the domain. Defaults to 1.1.
            dt_factor (float, optional): distance traveled by the probe per time step. Defaults to 1e-10.
            nt (int, optional): maximum number of time steps. Defaults to int(1e6).
            log_all (bool, optional): whether to log field variables. Defaults to False.
            pct_log (float, optional): defines frequency of console logging as a percent of the total estimated number of steps. Defaults to 0.1.

        Raises:
            ValueError: if the target probe position is less than -R/10, the simulation will not be valid
            ValueError: if the moduli are non-physical, i.e. Gg < Ge or negative
        """
        super().__init__(Gg, Ge, Tau, poissons, R, v_tip, h0, h0_target, p_func, *args, nr=nr, dr_factor=dr_factor, dt_factor=dt_factor, nt=nt, log_all=log_all, pct_log=pct_log)
        # get solid parameters for the iterative solver protocol
        if Ge == 0 and Tau == 0:  # elastic case
            self.b1 = 0
            self.b0 = Gg
            self.c1 = 0
            self.c0 = (1 - poissons ** 2)
        elif Ge > 0 and Tau > 0 and Ge <= Gg:  # viscoelastic case
            self.b1 = Gg * Tau
            self.b0 = Ge
            self.c1 = (1 - poissons ** 2) * Tau
            self.c0 = (1 - poissons ** 2)
        else:  # invalid case
            raise ValueError('invalid moduli')
        
        self.eta = eta
        self.solution.update({'eta': eta})
        
        # self explanatory warning
        if self.r[-1] < self.R * (1 / 5) ** (1 / 2):
            print('there might be an issue when indenting due to the brenner asymptotic correction')
    
    def make_loggers(self, state):
        """make loggers for the simulation

        Args:
            state (numpy array): state vector of the iterator
        """
        self.force = np.zeros(self.nt)
        self.separation = np.zeros(self.nt)
        self.position = np.zeros(self.nt)
        self.deformation = np.zeros(self.nt)
        self.time = np.zeros(self.nt)
        self.errors = np.zeros((self.nt, state.shape[0]))
        self.iters = np.zeros(self.nt)
    
    def make_field_loggers(self):
        """make loggers for the field variables
        """
        self.U = np.zeros((self.nt, self.r.size))
        self.H = np.zeros((self.nt, self.r.size))
        self.P = np.zeros((self.nt, self.r.size))
        self.P_fluid = np.zeros((self.nt, self.r.size))
    
    def calc_fluid_pressure(self, h0, h, h_dot):
        """calculate the fluid pressure by rearranging the Reynolds equation and integrating twice, two boundary conditions are required
        the first is due to the symmetry of the problem, the second is estimated using the brenner pressure solution as an estimate of the asymptotic
        behavior of the fluid pressure at r[-1], this could cause issues if the deformation is large. it improves the accuracy as the fluid pressure
        is fat-tailed, so if we enforced p_fluid(r[-1])=0, as is common, we would underpredict the pressure at large r

        Args:
            h0 (float): equilibrium separation between the probe and the surface
            h (numpy array): gap height
            h_dot (numpy array): gap velocity

        Returns:
            numpy array: pressure on the surface due to the fluid
        """
        # perform the first integration, using initial=0 as the integrating constant from the symmetry boundary condition of dp_fluid/dr = 0 at r=0
        interm1 = cumtrapz(12 * self.eta * self.r * h_dot, self.r, initial=0)
        # perform the second integration, arbitrarily set to start at 0, then add the asymptotic correction
        p_fluid = cumtrapz(interm1 / (self.r * h ** 3), self.r, initial=0)
        # using the typical method of implementing the boundary condition of p_fluid(inf)=0, we say p_fluid(r[-1])=0
        p_fluid -= p_fluid[-1]  # old way
        # add the asymptotic correction for p_fluid(r[-1])=p_brenner(r[-1])
        p_fluid += p_brenner(h0 + self.r[-1] ** 2 / (2 * self.R), self.v_tip, self.eta, self.R)[0]  # assuming p_brenner returns a tuple!!!!!!!!!!
        return p_fluid
    
    def calc_viscoelastic_deformation(self, u_prev, p, p_dot):
        """calculate the viscoelastic deformation using the iterative solver protocol,
        obtained by applying the viscoelastic-elastic correspondence principle to the Attard elastic equation (deformation of elastic half-space due to 
        arbitrary pressure) after first taking the equation into the frequency domain, then applying the arbitrary ODE definition of linear viscoelastic behavior
        and inserting it for the complex compliance (or complex modulus) of the material, then inverting back into the time domain to obtain the generalized ODE:
        b1 u_dot(t) + b0 u(t) = -c1 I(p_dot(t)) - c0 I(p(t))
        expanding the derivative with forward finite differences: u_dot(t)=(u(t+dt)-u(t))/dt
        inserting into the ODE and rearranging gives:
        u(t+dt) = (-c1 I(p_dot(t)) - c0 I(p(t)) + b1 u(t)/dt) / (b0 + b1/dt)

        Args:
            u_prev (numpy array): previous deformation
            p (numpy array): pressure on the surface
            p_dot (numpy array): time derivative of the pressure on the surface

        Returns:
            numpy array: viscoealstic deformation at the next step
        """
        I0 = (p * self.r) @ self.k_ij * self.dr
        I1 = (p_dot * self.r) @ self.k_ij * self.dr
        return (-self.c1 * I1 - self.c0 * I0 + self.b1 * u_prev / self.dt) / (self.b0 + self.b1 / self.dt)
    
    def brenner_asymptotic_force_finite_domain_correction(self, h0):
        """order of magnitude correction factor for the part of the domain for r[-1] to infinity, obtained by assuming the deformation is 0 beyond r[-1], 
        and integrating the brenner pressure from r[-1] to infinity. this can cause issues if the probe is indented beyond (1/5)^(1/2) * R, 
        it is useful as the pressure is fat-tailed, so if we enforced p_fluid(r[-1])=0, as is common, we would lose the force contribution from the fluid at large r

        Args:
            h0 (float): equilibrium separation between the probe and the surface

        Returns:
            float: force correction
        """
        # obtained by assuming deformation is 0 at r_max and solving for the pressure using the
        # brenner solution for the fluid pressure and then integrating the brenner pressure
        # from r_max to infinity in the force integral to obtain the correction
        return -12 * self.eta * self.v_tip * self.R ** 3 / (self.r[-1] ** 2 + 2 * h0 * self.R)
    
    def viscoelastic_fluid_iterator(self, state, state_n_guess, h0, h0_n, err_tol=1e-10, max_iter=100, wf=0.5):
        """use iteration to solve the coupled viscoelastic and fluid field equations
        n denotes the value calculated at the next time step i.e. t+dt
        
        Args:
            state (numpy array of numpy arrays): u, u_dot, h, h_dot, p_fluid, p_fluid_dot, p_surface, p_surface_dot
            state_n_guess (numpy array of numpy arrays): the guess for the state vector at the next time step
            h0 (float): current position of the probe
            h0_n (float): position of the probe at time t+dt
            err_tol (float, optional): convergence tolerance for iteration. Defaults to 1e-10.
            max_iter (int, optional): maximum number of convergence iterations. Defaults to 100.
            wf (float, optional): mixing ratio within (0,1). values closer to 1 gives slower convergence
            but more accurate results. Defaults to 0.5.
        Returns:
            (numpy array of numpy arrays, numpy array, int): 
            (u, u_dot, h, h_dot, p_fluid, p_fluid_dot, p_surface, p_surface_dot at the next time step t+dt),
            (convergence error at final iteration),
            (number of iterations)
        """
        state_n_guess_new = state_n_guess.copy()
        u, u_dot, h, h_dot, p_fluid, p_fluid_dot, p_surface, p_surface_dot = state  # a lot of these are not used, but are here for clarity and future use
        err = np.ones(state.shape[0]) * np.inf
        num_iter = 0
        while num_iter < max_iter and np.any(err > err_tol):
            num_iter += 1
            # get the values from the state guess
            u_n, u_n_dot, h_n, h_n_dot, p_fluid_n, p_fluid_dot_n, p_surface_n, p_surface_dot_n = state_n_guess
            # calculate the new fluid pressure using the guess for h and h_dot at the next time step
            p_fluid_n_new = self.calc_fluid_pressure(self.h0, h_n, h_n_dot)
            # calculate the new fluid pressure derivative
            p_fluid_dot_n_new = (p_fluid_n_new - p_fluid) / self.dt
            # calculate the new surface pressure using the guess for h and h_dot at the next time step
            p_surface_n_new = self.p_func(h_n, *self.args)[0]  # THIS IS ASSUMING P_FUNC RETURNS A TUPLE!!!!!!!!!!!!!!
            # calculate the new surface pressure derivative
            p_surface_dot_n_new = (p_surface_n_new - p_surface) / self.dt
            # calculate the new total pressure on the surface and its derivative
            p_n_new = p_fluid_n_new + p_surface_n_new
            p_dot_n_new = p_fluid_dot_n_new + p_surface_dot_n_new
            # calculate the new deformation of the elastic surface
            u_n_new = self.calc_viscoelastic_deformation(u, p_n_new, p_dot_n_new)
            # calculate the new deformation rate
            u_dot_n_new = (u_n_new - u) / self.dt
            # calculate the new gap
            h_n_new = h0_n + self.r ** 2 / (2 * self.R) - u_n_new
            # calculate the new gap rate
            h_dot_n_new = (h0_n - h0) / self.dt - u_dot_n_new
            # create the new state
            state_n_guess_new = np.array([u_n_new, u_dot_n_new, h_n_new, h_dot_n_new, p_fluid_n_new, p_fluid_dot_n_new, p_surface_n_new, p_surface_dot_n_new])
            # determine the convergence
            err = np.linalg.norm(state_n_guess_new - state_n_guess, axis=1)
            # update the state guess using mixing rule
            state_n_guess = wf * state_n_guess + (1 - wf) * state_n_guess_new
        return state_n_guess, err, num_iter
    
    def solve(self, wf=0.5, err_tol=1e-10, max_iter=100):
        """explicitly solve the coupled viscoelastic and fluid field equations and generate a force curve

        Args:
            wf (float, optional): mixing ratio for the iteration update, larger is more accurate but slower. Defaults to 0.5.
            err_tol (float, optional): convergence tolerance for the iteration. Defaults to 1e-10.
            max_iter (int, optional): maximum number of iterations, larger is more accurate but slower. Defaults to 100.

        Returns:
            dict: solution dict containing all the relevant information
        """
        # log fields if necessary
        if self.log_all:
            self.make_field_loggers()
        # initialize starting state
        u = np.zeros(self.r.shape)
        h = self.calculate_gap(u, self.h0)
        p_fluid, p_fluid_h = p_brenner(h, self.v_tip, self.eta, self.R)
        p_surface, p_surface_h = self.p_func(h, *self.args)
        state = np.array([u, u * 0, h, self.v_tip + u * 0, p_fluid, p_fluid_h * self.v_tip, p_surface, p_surface_h * self.v_tip])
        # initialize previous state
        h_prev = self.calculate_gap(u, self.h0 - self.v_tip * self.dt)
        p_fluid, p_fluid_h = p_brenner(h_prev, self.v_tip, self.eta, self.R)
        p_surface, p_surface_h = self.p_func(h_prev, *self.args)
        state_prev = np.array([u, u * 0, h_prev, self.v_tip + u * 0, p_fluid, p_fluid_h * self.v_tip, p_surface, p_surface_h * self.v_tip])
        # make loggers
        self.make_loggers(state)
        # loop
        for i in range(self.nt):
            # use the update rule to estimate the next state (it is done by equating forward
            # and backward differences at a fixed time and rearranging to find state(t+dt)
            # due to state(t) and state(t-dt))
            state_next_guess = 2 * state - state_prev
            # iterate to refine the guess of the next state
            state_next, err, n_iter = self.viscoelastic_fluid_iterator(state, state_next_guess, self.h0, self.h0 + self.v_tip * self.dt, err_tol, max_iter, wf)
            # update the previous state
            state_prev = state.copy()
            state = state_next.copy()
            # calculate observables
            total_pressure = state[6] + state[4]
            self.force[i] = self.calculate_force(total_pressure)
            # this is a small correction (about an order of magnitude) but it could introduce
            # an error when h0 is negative and h0 * R * 2 passes beyond r_max
            # if the maximum indentation depth (min value of h0) is limited to -R/10, then this gives
            # a criterion of r_max > ~R/2 which is ok
            self.force[i] += self.brenner_asymptotic_force_finite_domain_correction(self.h0)
            self.deformation[i] = state[0][0]
            self.position[i] = self.h0 + self.v_tip * self.dt
            self.separation[i] = state[2][0]
            self.errors[i] = err
            self.iters[i] = n_iter
            self.time[i] = i * self.dt
            # log to console
            self.console_log(i)
            # log fields if necessary
            if self.log_all:
                self.U[i] = state[0]
                self.H[i] = state[2]
                self.P[i] = state[6] + state[4]
                self.P_fluid[i] = state[4]
            # advance the probe for the next time step
            self.h0 += self.v_tip * self.dt
            # early stopping conditions
            if self.h0 <= self.h0_target:
                self.force = self.force[:i]
                self.deformation = self.deformation[:i]
                self.position = self.position[:i]
                self.separation = self.separation[:i]
                self.time = self.time[:i]
                self.errors = self.errors[:i]
                self.iters = self.iters[:i]
                if self.log_all:
                    self.U = self.U[:i]
                    self.H = self.H[:i]
                    self.P = self.P[:i]
                    self.P_fluid = self.P_fluid[:i]
                break
        # return solution data
        sol_df = pd.DataFrame({'time': self.time, 'force': self.force, 'deformation': self.deformation, 'position': self.position, 'separation': self.separation, 
                               'iterations': self.iters, 'conv_u': self.errors[:, 0], 'conv_u_dot': self.errors[:, 1], 'conv_h': self.errors[:, 2],
                               'conv_h_dot': self.errors[:, 3], 'conv_p_fluid': self.errors[:, 4], 'conv_p_fluid_dot': self.errors[:, 5],
                               'conv_p_surface': self.errors[:, 6], 'conv_p_surface_dot': self.errors[:, 7]})
        self.solution['data'] = sol_df
        if self.log_all:
            self.solution['field_variables'] = {'U': self.U, 'H': self.H, 'P': self.P, 'P_fluid': self.P_fluid, 'r': self.r, 't': self.time}
        print('done! returning solution, it is also saved as self.solution')
        return self.solution
    
     
class ContactSimOde(ContactSim):
    
    def __init__(self, Gg, Ge, Tau, poissons, R, v_tip, h0, h0_target, p_func, *args, nr=int(1e3), dr_factor=1.1, dt_factor=1e-10, nt=int(1e6), log_all=False, pct_log=0.1):
        """solve the solid PDE using an implicit time stepping routine for the itneraction of a rigid, spherical probe with a semi-infinite,
        linear elastic or linear viscoelastic half-space.  the scheme is based on a generalization of Attard's equation and Rajabifar's solution
        The primary assumption is that the initial separation and the maximum deformation are smaller than the probe radius, so that the Derjaguin
        approximation is valid for the surface forces and the lubrication approximation (Reynolds equation) is valid for the fluid forces.  Both of these
        allow the force on the probe to be calculated as the integral of the pressure on the surface

        Args:
            Gg (float): instantaneous (glassy) shear modulus of the surface
            Ge (float): equilibrium (rubbery) shear modulus of the surface
            Tau (float): relaxation time of the surface
            poissons (float): poissons ratio of the surface
            R (float): radius of the probe
            v_tip (float): tip velocity
            h0 (float): separation between the probe and the equilibrium position of the surface at r=0 (i.e. global position of the tip of the probe)
            h0_target (float): target position of the tip of the probe
            p_func (_type_): function used to calculate the pressure on the surface with optional args, must be of the form p_func(h, *args) and return a tuple
            of the pressure and the derivative of the pressure with respect to the gap height
            nr (int, optional): number of points in the domain. Defaults to int(1e3).
            dr_factor (float, optional): this multiplied with the probe radius gives the maximum radial distance in the domain. Defaults to 1.1.
            dt_factor (float, optional): distance traveled by the probe per time step. Defaults to 1e-10.
            nt (int, optional): maximum number of time steps. Defaults to int(1e6).
            log_all (bool, optional): whether to log field variables. Defaults to False.
            pct_log (float, optional): defines frequency of console logging as a percent of the total estimated number of steps. Defaults to 0.1.

        Raises:
            ValueError: if the target probe position is less than -R/10, the simulation will not be valid
            ValueError: if the moduli are non-physical, i.e. Gg < Ge or negative
        """        
        super().__init__(Gg, Ge, Tau, poissons, R, v_tip, h0, h0_target, p_func, *args, nr=nr, dr_factor=dr_factor, dt_factor=dt_factor, nt=nt, log_all=log_all, pct_log=pct_log)
        # get solid parameters for the ode solver protocol
        if Ge == 0 and Tau == 0:  # elastic case
            self.b1 = Gg
            self.b0 = 0
            self.c1 = (1 - poissons ** 2)
            self.c0 = 0
        elif Ge > 0 and Tau > 0 and Ge <= Gg:  # viscoelastic case
            self.b1 = Gg * Tau
            self.b0 = Ge
            self.c1 = (1 - poissons ** 2) * Tau
            self.c0 = (1 - poissons ** 2)
        else:  # who knows?
            raise ValueError('invalid moduli')
        
        # define the identity matrix for use as dirac delta kernel
        self.I = np.eye(self.r.size)

    def rhs(self, state):
        """calculate the right hand side of the solid ODE

        Args:
            state (numpy array): deformation and separation between the probe and the equilibrium position of the surface at r=0 (i.e. global position of the tip of the probe)

        Returns:
            numpy array: derivative of state vector (deformation rate, probe velocity)
        """
        # unpack state
        u, h0 = state
        # calculate gap
        h = self.calculate_gap(u, h0)
        # calculate pressure and derivative
        p, p_h = self.p_func(h, *self.args)
        # calculate lhs integral operator as a matrix
        lhs = self.c1 * self.dr * self.k_ij * p_h * self.r - self.b1 * self.I
        # calculate the rhs source term
        rhs = self.c1 * self.v_tip * self.dr * (p_h * self.r) @ self.k_ij + self.c0 * self.dr * (p * self.r) @ self.k_ij + self.b0 * u
        # invert the equation to obtain the derivative of u
        u_dot = np.linalg.solve(lhs, rhs)
        return np.array([u_dot, self.v_tip], dtype=object), (p, h)
    
    def rk4(self, state):
        """4th order Runge-Kutta time stepping scheme

        Args:
            state (numpy array): state vector (deformation, separation in our case)

        Returns:
            numpy array: state vector derivative (deformation rate, probe velocity in our case), extra values from rhs calculation (pressure, gap in our case)
        """
        k1, _ = self.rhs(state)
        k2, _ = self.rhs(state + k1 * self.dt / 2)
        k3, _ = self.rhs(state + k2 * self.dt / 2)
        k4, extra = self.rhs(state + k3 * self.dt)
        return state + (k1 + 2 * k2 + 2 * k3 + k4) * self.dt / 6, extra
    
    def solve(self, h_target=0, u_target=np.inf):
        """solve the solid ODE using the implicit time stepping routine

        Returns:
            dict: solution dict containing all the relevant information
        """
        # make loggers
        self.make_loggers()
        # log fields if necessary
        if self.log_all:
            self.make_field_loggers()
        # make solid deformation field
        u = np.zeros(self.r.size)
        # make state
        state = np.array([u, self.h0], dtype=object)
        # loop
        for i in range(self.nt):
            # update state
            state, (p, h) = self.rk4(state)
            u, self.h0 = state
            # calculate observables
            self.force[i] = self.calculate_force(p)
            self.deformation[i] = u[0]
            self.position[i] = self.h0
            self.separation[i] = h[0]
            self.time[i] = i * self.dt
            # log to console
            self.console_log(i)
            # log fields if necessary
            if self.log_all:
                self.U[i] = u
                self.H[i] = h
                self.P[i] = p
            # early stopping conditions
            if self.h0 <= self.h0_target or np.any(h <= h_target) or np.any(abs(u) >= abs(u_target)):
                self.force = self.force[:i]
                self.deformation = self.deformation[:i]
                self.position = self.position[:i]
                self.separation = self.separation[:i]
                self.time = self.time[:i]
                if self.log_all:
                    self.U = self.U[:i]
                    self.H = self.H[:i]
                    self.P = self.P[:i]
                break
        # return solution data
        sol_df = pd.DataFrame({'time': self.time, 'force': self.force, 'deformation': self.deformation, 'position': self.position, 'separation': self.separation})
        self.solution['data'] = sol_df
        if self.log_all:
            self.solution['field_variables'] = {'U': self.U, 'H': self.H, 'P': self.P, 'r': self.r, 't': self.time}
        print('done! returning solution, it is also saved as self.solution')
        return self.solution
    
class ContactSimOdeFluid(ContactSimOde):
    
    def __init__(self, Gg, Ge, Tau, poissons, R, v_tip, h0, h0_target, eta, p_func, *args, nr=int(1e3), dr_factor=1.1, step_size=1e-10, nt=int(1e6), log_all=False, pct_log=0.1, h_target=0, u_target=np.inf):
        """solve the solid PDE using an implicit time stepping routine for the itneraction of a rigid, spherical probe with a semi-infinite,
        linear elastic or linear viscoelastic half-space.  the scheme is based on a generalization of Attard's equation and Rajabifar's solution
        The primary assumption is that the initial separation and the maximum deformation are smaller than the probe radius, so that the Derjaguin
        approximation is valid for the surface forces and the lubrication approximation (Reynolds equation) is valid for the fluid forces.  Both of these
        allow the force on the probe to be calculated as the integral of the pressure on the surface

        Args:
            Gg (float): instantaneous (glassy) shear modulus of the surface
            Ge (float): equilibrium (rubbery) shear modulus of the surface
            Tau (float): relaxation time of the surface
            poissons (float): poissons ratio of the surface
            R (float): radius of the probe
            v_tip (float): tip velocity
            h0 (float): separation between the probe and the equilibrium position of the surface at r=0 (i.e. global position of the tip of the probe)
            h0_target (float): target position of the tip of the probe
            p_func (_type_): function used to calculate the pressure on the surface with optional args, must be of the form p_func(h, *args) and return a tuple
            of the pressure and the derivative of the pressure with respect to the gap height
            nr (int, optional): number of points in the domain. Defaults to int(1e3).
            dr_factor (float, optional): this multiplied with the probe radius gives the maximum radial distance in the domain. Defaults to 1.1.
            step_size (float, optional): distance traveled by the probe per time step. Defaults to 1e-10.
            nt (int, optional): maximum number of time steps. Defaults to int(1e6).
            log_all (bool, optional): whether to log field variables. Defaults to False.
            pct_log (float, optional): defines frequency of console logging as a percent of the total estimated number of steps. Defaults to 0.1.
            h_target (float, optional): defines the target separation between the probe and the surface. Defaults to 0.
            u_target (float, optional): defines the target deformation of the probe. Defaults to np.inf.

        Raises:
            ValueError: if the target probe position is less than -R/10, the simulation will not be valid
            ValueError: if the moduli are non-physical, i.e. Gg < Ge or negative
        """        
        super().__init__(Gg, Ge, Tau, poissons, R, v_tip, h0, h0_target, p_func, *args, nr=nr, dr_factor=dr_factor, dt_factor=step_size, nt=nt, log_all=log_all, pct_log=pct_log)
        self.h_target = h_target
        self.u_target = u_target
        # define the fluid viscosity
        self.eta = eta
        self.solution['eta'] = self.eta
        # making vectors
        self.v_tip = np.ones(self.r.shape) * self.v_tip
        self.h0 = np.ones(self.r.shape) * self.h0
        # self explanatory warning
        if self.r[-1] < self.R * (1 / 5) ** (1 / 2):
            print('there might be an issue when indenting due to the brenner asymptotic correction')

    def console_log(self, step):
        """log the simulation to the console at the specified step

        Args:
            step (int): current simulation step
        """
        if step % int((self.h0_init - self.h0_target) / (self.v_tip[0] * self.dt) * self.pct_log) == 0:
            print('step: {} | h0: {:.1f} (nm) | h0_target: {:.1f} (nm) | force: {:.1f} (nN) | separation: {:.1f} (nm) | time: {:.3f} (s)'.format(step, self.h0[0] * 1e9, self.h0_target * 1e9, self.force[step] * 1e9, self.separation[step] * 1e9, self.time[step]))
  

    def d1(self, f):
        """first derivative of f, using a 2nd order central difference scheme

        Args:
            f (np.array): some array

        Returns:
            np.array: first derivative of f
        """
        return (f[2:] - f[:-2]) / (2 * self.dr)

    def calc_fluid_pressure(self, h0, h, h_dot):
        """calculate the fluid pressure by rearranging the Reynolds equation and integrating twice, two boundary conditions are required
        the first is due to the symmetry of the problem, the second is estimated using the brenner pressure solution as an estimate of the asymptotic
        behavior of the fluid pressure at r[-1], this could cause issues if the deformation is large. it improves the accuracy as the fluid pressure
        is fat-tailed, so if we enforced p_fluid(r[-1])=0, as is common, we would underpredict the pressure at large r

        Args:
            h0 (float): equilibrium separation between the probe and the surface
            h (numpy array): gap height
            h_dot (numpy array): gap velocity

        Returns:
            numpy array: pressure on the surface due to the fluid
        """
        # perform the first integration, using initial=0 as the integrating constant from the symmetry boundary condition of dp_fluid/dr = 0 at r=0
        interm1 = cumtrapz(12 * self.eta * self.r * h_dot, self.r, initial=0)
        # perform the second integration, arbitrarily set to start at 0, then add the asymptotic correction
        p_fluid = cumtrapz(interm1 / (self.r * h ** 3), self.r, initial=0)
        # using the typical method of implementing the boundary condition of p_fluid(inf)=0, we say p_fluid(r[-1])=0
        p_fluid -= p_fluid[-1]  # old way
        # add the asymptotic correction for p_fluid(r[-1])=p_brenner(r[-1])
        p_fluid += p_brenner(h0 + self.r[-1] ** 2 / (2 * self.R), self.v_tip, self.eta, self.R)[0]  # assuming p_brenner returns a tuple!!!!!!!!!!
        return p_fluid
    
    
    def calc_fluid_pressure_gradient(self, p_fluid, h):
        """calculate the gradient of the fluid pressure by differentiating the reynolds equation by h, rearranging, and integrating once

        Args:
            p_fluid (numpy array): pressure on the surface due to the fluid
            h (numpy array): gap height

        Returns:
            numpy array: gradient of the fluid pressure
        """
        p_fluid_h_numerical = np.zeros(self.r.shape)
        p_fluid_h_numerical[1:-1] = -3 * cumtrapz(self.d1(p_fluid) / h[1:-1], self.r[1:-1], initial=0)  # symmetric boundary condition
        p_fluid_h_numerical[1:-1] -= p_fluid_h_numerical[-2]  # we use -2 and not -1, bc -1 is default 0 due to the .zeros defintion above
        p_fluid_h_numerical[1:-1] += p_brenner(h, self.v_tip, self.eta, self.R)[1][-1]  # apply brenner pressure asymptote
        # apply symmetry boundary condition (zero gradient at r=0)
        p_fluid_h_numerical[0] = p_fluid_h_numerical[1]
        # use splines to interpolate the boundary at large r
        spline = splrep(self.r[:-1], p_fluid_h_numerical[:-1])
        p_fluid_h_numerical[-1] = splev(self.r[-1], spline)
        return p_fluid_h_numerical
    
    def brenner_asymptotic_force_finite_domain_correction(self, h0):
        """order of magnitude correction factor for the part of the domain for r[-1] to infinity, obtained by assuming the deformation is 0 beyond r[-1], 
        and integrating the brenner pressure from r[-1] to infinity. this can cause issues if the probe is indented beyond (1/5)^(1/2) * R, 
        it is useful as the pressure is fat-tailed, so if we enforced p_fluid(r[-1])=0, as is common, we would lose the force contribution from the fluid at large r

        Args:
            h0 (float): equilibrium separation between the probe and the surface

        Returns:
            float: force correction
        """
        # obtained by assuming deformation is 0 at r_max and solving for the pressure using the
        # brenner solution for the fluid pressure and then integrating the brenner pressure
        # from r_max to infinity in the force integral to obtain the correction
        return -12 * self.eta * self.v_tip * self.R ** 3 / (self.r[-1] ** 2 + 2 * h0 * self.R)
    
    
    def rhs_elastic_iter(self, u, h0, h_prev, dt, E_star, max_iter=100, tol=1e-6, wf=1):
        h0 = h0[0]
        u_prev = u.copy()
        h = self.calculate_gap(u, h0)
        p_surface, _ = self.p_func(h, *self.args)
        h_dot = (h - h_prev) / dt
        p_fluid = self.calc_fluid_pressure(self.h0, h, h_dot)
        p_total = p_surface + p_fluid
        num_iters = 0
        err = np.inf
        while num_iters < max_iter and err > tol:
            num_iters += 1
            u_new = - 1 / E_star * self.dr * self.k_ij @ (p_total * self.r)
            u_dot = (u_new - u_prev) / dt
            h_dot = (self.v_tip - u_dot) * wf + (1 - wf) * h_dot  # weighted average of the old and new h_dot
            p_fluid = self.calc_fluid_pressure(self.h0, h, h_dot)
            p_total = p_surface + p_fluid
            err = np.mean(abs(u_new - u_prev))
            u_prev = u_new.copy()
        return u_new, p_total, p_fluid, p_surface, num_iters, err
            
            
    
    def rhs_semi_iter(self, state, h_prev, dt, iterate=False, max_iter=100, tol=1e-6, wf=1):
        """calculate the right hand side of the solid ODE

        Args:
            state (numpy array): deformation and separation between the probe and the equilibrium position of the surface at r=0 (i.e. global position of the tip of the probe)

        Returns:
            numpy array: derivative of state vector (deformation rate, probe velocity)
        """
        # unpack state
        u, h0 = state
        h0 = h0[0]
        # calculate gap
        h = self.calculate_gap(u, h0)
        # calculate surface pressure and derivative
        p, p_h = self.p_func(h, *self.args)
        # use estimate for h_dot to get the fluid pressure
        h_dot = (h - h_prev) / dt
        # calculate the fluid pressure
        p_fluid = self.calc_fluid_pressure(self.h0, h, h_dot)
        # calculate the fluid pressure gradient
        p_fluid_h = self.calc_fluid_pressure_gradient(p_fluid, h)
        # determine total pressure
        p_total = p + p_fluid
        p_h_total = p_h + p_fluid_h
        # begin iteration loop
        num_iters = 0
        err = np.inf
        while num_iters < max_iter and err > tol:
            # calculate lhs integral operator as a matrix
            lhs = self.c1 * self.dr * self.k_ij * p_h_total * self.r - self.b1 * self.I
            # calculate the rhs source term
            rhs = self.c1 * self.v_tip * self.dr * (p_h_total * self.r) @ self.k_ij + self.c0 * self.dr * (p_total * self.r) @ self.k_ij + self.b0 * u
            # invert the equation to obtain the derivative of u
            u_dot = np.linalg.solve(lhs, rhs)
            # iterate if needed
            if not iterate:
                break
            num_iters += 1
            # calculate the new h_dot
            h_dot = (self.v_tip - u_dot) * wf + (1 - wf) * h_dot  # weighted average of the old and new h_dot
            # calculate the fluid pressure
            p_fluid_new = self.calc_fluid_pressure(self.h0, h, h_dot)
            # calculate the fluid pressure gradient
            p_fluid_h_new = self.calc_fluid_pressure_gradient(p_fluid, h)
            # calculate the error
            err = np.linalg.norm(p_fluid_new - p_fluid) / np.linalg.norm(p_fluid) + np.linalg.norm(p_fluid_h_new - p_fluid_h) / np.linalg.norm(p_fluid_h)
            p_fluid = p_fluid_new.copy()
            p_fluid_h = p_fluid_h_new.copy()
            # determine total pressure
            p_total = p + p_fluid
            p_h_total = p_h + p_fluid_h
        return np.array([u_dot, self.v_tip]), (h_dot, num_iters, err)
    
    
    def rk4_v2(self, state, h_prev, iterate=False, max_iter=100, tol=1e-6, wf=1):
        """4th order Runge-Kutta time stepping scheme

        Args:
            state (numpy array): state vector (deformation, separation in our case)

        Returns:
            numpy array: state vector derivative (deformation rate, probe velocity in our case), extra values from rhs calculation (pressure, gap in our case)
        """
        k1, (h_dot1, i1, e1) = self.rhs_semi_iter(state, h_prev, self.dt, iterate, max_iter, tol, wf)
        k2, (h_dot2, i2, e2) = self.rhs_semi_iter(state + k1 * self.dt / 2, h_prev, self.dt / 2, iterate, max_iter, tol, wf)
        k3, (h_dot3, i3, e3) = self.rhs_semi_iter(state + k2 * self.dt / 2, h_prev, self.dt / 2, iterate, max_iter, tol, wf)
        k4, (h_dot4, i4, e4) = self.rhs_semi_iter(state + k3 * self.dt, h_prev, self.dt, iterate, max_iter, tol, wf)
        return state + (k1 + 2 * k2 + 2 * k3 + k4) * self.dt / 6, ((h_dot1 + h_dot2 * 2 + h_dot3 * 2 + h_dot4) / 6, i1 + i2 * 2 + i3 * 2 + i4, e1 + e2 * 2 + e3 * 2 + e4)

    
    def solve_elastic(self, E_star, max_iter=100, tol=1e-6, wf=1, finite_domain_correction=False):
        # make loggers
        self.make_loggers()
        self.force_fluid = np.zeros(self.nt)
        self.force_surface = np.zeros(self.nt)
        self.conv_iters = np.zeros(self.nt)
        self.conv_error = np.zeros(self.nt)
        # log fields if necessary
        if self.log_all:
            self.make_field_loggers()
            self.P_surface = np.zeros((self.nt, self.r.size))
            self.P_fluid = np.zeros((self.nt, self.r.size))
        # make solid deformation field
        u = np.zeros(self.r.size)
        h = self.calculate_gap(u, self.h0)
        for i in range(self.nt):
            h_prev = h.copy()
            u, p_total, p_fluid, p_surface, num_iters, err = self.rhs_elastic_iter(u, self.h0, h_prev, self.dt, E_star, max_iter, tol, wf)
            h = self.calculate_gap(u, self.h0)
            h_prev = h.copy()
            self.h0 += self.v_tip * self.dt
            # calculate observables
            self.force[i] = self.calculate_force(p_total)
            self.force_fluid[i] = self.calculate_force(p_fluid)
            self.force_surface[i] = self.calculate_force(p_surface)
            # calculate finite domain fluid pressure correction if needed
            if finite_domain_correction:
                fluid_force_correction = self.brenner_asymptotic_force_finite_domain_correction(self.h0)[0]
                self.force_fluid[i] += fluid_force_correction
                self.force[i] += fluid_force_correction
            self.conv_iters[i] = num_iters
            self.conv_error[i] = err
            self.deformation[i] = u[0]
            self.position[i] = self.h0[0]
            self.separation[i] = h[0]
            self.time[i] = i * self.dt
            # log to console
            self.console_log(i)
            # log fields if necessary
            if self.log_all:
                self.U[i] = u
                self.H[i] = h
                self.P[i] = p_total
                self.P_fluid[i] = p_fluid
                self.P_surface[i] = p_surface
            # early stopping conditions
            if self.h0[0] <= self.h0_target or any(h < self.h_target) or any(abs(u) > self.u_target) or self.conv_error[i] >= tol or self.conv_iters[i] >= max_iter or np.isnan(self.force[i]):
                self.force_fluid = self.force_fluid[:i]
                self.force_surface = self.force_surface[:i]
                self.conv_iters = self.conv_iters[:i]
                self.conv_error = self.conv_error[:i]
                self.force = self.force[:i]
                self.deformation = self.deformation[:i]
                self.position = self.position[:i]
                self.separation = self.separation[:i]
                self.time = self.time[:i]
                if self.log_all:
                    self.U = self.U[:i]
                    self.H = self.H[:i]
                    self.P = self.P[:i]
                    self.P_fluid = self.P_fluid[:i]
                    self.P_surface = self.P_surface[:i]
                break
        # return solution data
        sol_df = pd.DataFrame({'time': self.time, 'force': self.force, 'deformation': self.deformation, 'position': self.position, 'separation': self.separation,
                               'force_fluid': self.force_fluid, 'force_surface': self.force_surface, 'conv_iters': self.conv_iters, 'conv_error': self.conv_error})
        self.solution['data'] = sol_df
        if self.log_all:
            self.solution['field_variables'] = {'U': self.U, 'H': self.H, 'P': self.P, 'r': self.r, 't': self.time, 'P_fluid': self.P_fluid, 'P_surface': self.P_surface}
        print('done! returning solution, it is also saved as self.solution')
        return self.solution

    
    def simple_elastic_solve(self, E, max_iter=10000, tol=1e-11, wf=0.5, finite_domain_correction=True):
        """solve the solid ODE using the explicit time stepping routine

        Returns:
            dict: solution dict containing all the relevant information
        """
        # make loggers
        self.make_loggers()
        self.force_fluid = np.zeros(self.nt)
        self.force_surface = np.zeros(self.nt)
        self.conv_iters = np.zeros(self.nt)
        self.conv_error = np.zeros(self.nt)
        # log fields if necessary
        if self.log_all:
            self.make_field_loggers()
            self.P_surface = np.zeros((self.nt, self.r.size))
            self.P_fluid = np.zeros((self.nt, self.r.size))
        # make solid deformation field
        u = np.zeros(self.r.size)
        h = self.calculate_gap(u, self.h0)
        h_dot = np.zeros(self.r.size) + self.v_tip
        v_tip = self.v_tip[0]
        eta = self.eta
        r = self.r.copy()
        k_ij = self.k_ij.copy()
        R = self.R
        dr = self.dr
        dt = self.dt
        
        def compute_p(h0, h, h_dot, r, eta, v_tip, R):
            interm1 = cumtrapz(12 * eta * r * h_dot, r, initial=0)
            p_fluid = cumtrapz(interm1 / (r * h ** 3), r, initial=0)
            p_fluid -= p_fluid[-1]  # old way
            p_fluid += p_brenner(h0 + r[-1] ** 2 / (2 * R), v_tip, eta, R)[0]
            return p_fluid

        for i in range(self.nt):
            # iterate
            for _ in range(max_iter):
                h = self.h0 + r ** 2 / (2 * R) - u
                p = compute_p(self.h0, h, h_dot, r, eta, v_tip, R)
                u_new = - (1 - 0.5 ** 2) / E * dr * k_ij @ (p * r)
                u_dot = (u_new - u) / dt
                h_dot_new = v_tip - u_dot
                err = np.linalg.norm(h_dot - h_dot_new)
                if err < tol:
                    h_dot = h_dot_new
                    break
                h_dot = (1 - wf) * h_dot + wf * h_dot_new  # underrelaxation step
                converged = True
            else:
                converged = False
            
            # Euler's time-stepping
            h += dt * h_dot
            u += dt * u_dot
            self.h0 += dt * v_tip

            # calculate observables
            self.force[i] = self.calculate_force(p)
            self.force_fluid[i] = self.calculate_force(p)
            self.force_surface[i] = self.calculate_force(p * 0)
            # calculate finite domain fluid pressure correction if needed
            if finite_domain_correction:
                fluid_force_correction = self.brenner_asymptotic_force_finite_domain_correction(self.h0)[0]
                self.force_fluid[i] += fluid_force_correction
                self.force[i] += fluid_force_correction
            self.conv_iters[i] = _
            self.conv_error[i] = err
            self.deformation[i] = u[0]
            self.position[i] = self.h0[0]
            self.separation[i] = h[0]
            self.time[i] = i * self.dt
            # log to console
            self.console_log(i)
            # log fields if necessary
            if self.log_all:
                self.U[i] = u
                self.H[i] = h
                self.P[i] = p
                self.P_fluid[i] = p
                self.P_surface[i] = p * 0
            # early stopping conditions
            if self.h0[0] <= self.h0_target or any(h < self.h_target) or any(abs(u) > self.u_target):
                self.force_fluid = self.force_fluid[:i]
                self.force_surface = self.force_surface[:i]
                self.conv_iters = self.conv_iters[:i]
                self.conv_error = self.conv_error[:i]
                self.force = self.force[:i]
                self.deformation = self.deformation[:i]
                self.position = self.position[:i]
                self.separation = self.separation[:i]
                self.time = self.time[:i]
                if self.log_all:
                    self.U = self.U[:i]
                    self.H = self.H[:i]
                    self.P = self.P[:i]
                    self.P_fluid = self.P_fluid[:i]
                    self.P_surface = self.P_surface[:i]
                break
        # return solution data
        sol_df = pd.DataFrame({'time': self.time, 'force': self.force, 'deformation': self.deformation, 'position': self.position, 'separation': self.separation,
                               'force_fluid': self.force_fluid, 'force_surface': self.force_surface, 'conv_iters': self.conv_iters, 'conv_error': self.conv_error})
        self.solution['data'] = sol_df
        if self.log_all:
            self.solution['field_variables'] = {'U': self.U, 'H': self.H, 'P': self.P, 'r': self.r, 't': self.time, 'P_fluid': self.P_fluid, 'P_surface': self.P_surface}
        print('done! returning solution, it is also saved as self.solution')
        return self.solution
    
    
    def solve(self, iterate=False, max_iter=100, tol=1e-6, wf=1, finite_domain_correction=False, dynamic_dt=False, dynamic_dt_factor=1):
        """solve the solid ODE using the implicit time stepping routine

        Returns:
            dict: solution dict containing all the relevant information
        """
        # configure dynamic time step
        if dynamic_dt:
            self.dt_max = self.dt
            self.stability_factor = 1e3  # the product of max(p_h)/Gg * dt should be less than this
        # make loggers
        self.make_loggers()
        self.force_fluid = np.zeros(self.nt)
        self.force_surface = np.zeros(self.nt)
        self.conv_iters = np.zeros(self.nt)
        self.conv_error = np.zeros(self.nt)
        # log fields if necessary
        if self.log_all:
            self.make_field_loggers()
            self.P_surface = np.zeros((self.nt, self.r.size))
            self.P_fluid = np.zeros((self.nt, self.r.size))
        # make solid deformation field
        u = np.zeros(self.r.size)
        h = self.calculate_gap(u, self.h0)
        state = np.array([u, self.h0])

        for i in range(self.nt):
            # need to find good way to calculate h_dot for each rk4 step
            h_prev = h.copy()
            # perform update using rk4
            state, (h_dot, num_iters, err) = self.rk4_v2(state, h_prev, iterate=iterate, max_iter=max_iter, tol=tol, wf=wf)
            u, self.h0 = state
            # calculate gap
            h = self.calculate_gap(u, self.h0)
            # calculate surface pressure
            p, p_h = self.p_func(h, *self.args)
            # considering only the surface pressure gradient, adjust the timestep to promote stability
            if dynamic_dt:  # making dynamic_dt_factor smaller will make the simulation more stable
                self.dt = min(self.dt_max, 1e3 * self.b1 / max(p_h) * dynamic_dt_factor)
            # calculate fluid pressure
            p_fluid = self.calc_fluid_pressure(self.h0, h, h_dot)
            # determine total pressure
            p_total = p + p_fluid
            # calculate observables
            self.force[i] = self.calculate_force(p_total)
            self.force_fluid[i] = self.calculate_force(p_fluid)
            self.force_surface[i] = self.calculate_force(p)
            # calculate finite domain fluid pressure correction if needed
            if finite_domain_correction:
                fluid_force_correction = self.brenner_asymptotic_force_finite_domain_correction(self.h0)[0]
                self.force_fluid[i] += fluid_force_correction
                self.force[i] += fluid_force_correction
            self.conv_iters[i] = num_iters
            self.conv_error[i] = err
            self.deformation[i] = u[0]
            self.position[i] = self.h0[0]
            self.separation[i] = h[0]
            self.time[i] = i * self.dt
            # log to console
            self.console_log(i)
            # log fields if necessary
            if self.log_all:
                self.U[i] = u
                self.H[i] = h
                self.P[i] = p_total
                self.P_fluid[i] = p_fluid
                self.P_surface[i] = p
            # early stopping conditions
            if self.h0[0] <= self.h0_target or any(h < self.h_target) or any(abs(u) > self.u_target):
                self.force_fluid = self.force_fluid[:i]
                self.force_surface = self.force_surface[:i]
                self.conv_iters = self.conv_iters[:i]
                self.conv_error = self.conv_error[:i]
                self.force = self.force[:i]
                self.deformation = self.deformation[:i]
                self.position = self.position[:i]
                self.separation = self.separation[:i]
                self.time = self.time[:i]
                if self.log_all:
                    self.U = self.U[:i]
                    self.H = self.H[:i]
                    self.P = self.P[:i]
                    self.P_fluid = self.P_fluid[:i]
                    self.P_surface = self.P_surface[:i]
                break
        # return solution data
        sol_df = pd.DataFrame({'time': self.time, 'force': self.force, 'deformation': self.deformation, 'position': self.position, 'separation': self.separation,
                               'force_fluid': self.force_fluid, 'force_surface': self.force_surface, 'conv_iters': self.conv_iters, 'conv_error': self.conv_error})
        self.solution['data'] = sol_df
        if self.log_all:
            self.solution['field_variables'] = {'U': self.U, 'H': self.H, 'P': self.P, 'r': self.r, 't': self.time, 'P_fluid': self.P_fluid, 'P_surface': self.P_surface}
        print('done! returning solution, it is also saved as self.solution')
        return self.solution
    
class ContactSimOdeFluid2(ContactSimOdeFluid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def rhs_semi_iter(self, state, h_prev, dt, iterate=False, max_iter=100, tol=1e-6, wf=1):
        """calculate the right hand side of the solid ODE

        Args:
            state (numpy array): deformation and separation between the probe and the equilibrium position of the surface at r=0 (i.e. global position of the tip of the probe)

        Returns:
            numpy array: derivative of state vector (deformation rate, probe velocity)
        """
        # unpack state
        u, h0 = state
        h0 = h0[0]
        # calculate gap
        h = self.calculate_gap(u, h0)
        # calculate surface pressure and derivative
        p, p_h = self.p_func(h, *self.args)
        # use estimate for h_dot to get the fluid pressure
        h_dot = (h - h_prev) / dt
        # calculate the fluid pressure
        p_fluid = self.calc_fluid_pressure(self.h0, h, h_dot)
        # calculate the fluid pressure gradient
        p_fluid_h = self.calc_fluid_pressure_gradient(p_fluid, h)
        # determine total pressure
        p_total = p + p_fluid
        p_h_total = p_h + p_fluid_h
        # begin iteration loop
        num_iters = 0
        err = np.inf
        while num_iters < max_iter and err > tol:
            # calculate lhs integral operator as a matrix
            lhs = self.c1 * self.dr * self.k_ij * p_h_total * self.r - self.b1 * self.I
            # calculate the rhs source term
            rhs = self.c1 * self.v_tip * self.dr * (p_h_total * self.r) @ self.k_ij + self.c0 * self.dr * (p_total * self.r) @ self.k_ij + self.b0 * u
            # invert the equation to obtain the derivative of u
            u_dot = np.linalg.solve(lhs, rhs)
            # iterate if needed
            if not iterate:
                break
            num_iters += 1
            # calculate the new h_dot
            h_dot = (self.v_tip - u_dot) * wf + (1 - wf) * h_dot  # weighted average of the old and new h_dot
            # calculate the fluid pressure
            p_fluid_new = self.calc_fluid_pressure(self.h0, h, h_dot)
            # calculate the fluid pressure gradient
            p_fluid_h_new = self.calc_fluid_pressure_gradient(p_fluid, h)
            # calculate the error
            err = np.mean((p_fluid_new - p_fluid) / (6 * np.pi * abs(self.eta * self.v_tip[0] * self.R)))
            p_fluid = p_fluid_new.copy()
            p_fluid_h = p_fluid_h_new.copy()
            # determine total pressure
            p_total = p + p_fluid
            p_h_total = p_h + p_fluid_h
        return np.array([u_dot, self.v_tip]), (h_dot, num_iters, err)
    
