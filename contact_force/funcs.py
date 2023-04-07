import numpy as np


def brenner_force(separation, eta, v_tip, R):
    """calculate the brenner force model for a rigid surface

    Args:
        separation (float): distance between sphere and surface
        eta (float): fluid viscosity
        v_tip (float): tip velocity
        R (float): probe radius

    Returns:
        float: lubrication force
    """
    return -6 * np.pi * eta * v_tip * R ** 2 / separation


def hertzian(inden, G, v, R):
    '''
    calculate hertzian (elastic) contact force as a function of the indentation depth
    :param inden: float indentation depth
    :param G: float shear modulus
    :param v: float poissons ratio
    :param R: float indenter radius
    :return: float force
    '''
    return 4 / 3 * np.sqrt(R) * G / (1 - v ** 2) * inden ** 1.5

def ext_hertzian_hc(z, z0, G, v, R):
    '''
    calculate the extended hertzian (elastic) contact force, for the hard-contact case, as a function of the indenter
    position and sample equilibrium position
    :param z: float indenter position
    :param z0: float sample equilibrium position
    :param G: float sample shear modulus
    :param v: float sample poisson's ratio
    :param R: float indenter radius
    :return: float force
    '''
    z_shift = z - z0
    chi = (abs(z_shift) - z_shift) / 2
    return hertzian(chi, G, v, R)

def lee_radok(inden, v, R, Gg, Ge, Tau, t):
    '''
    calculate lee and radok (viscoelastic) contact force as a function of indentation depth
    :param inden: float indentation depth
    :param v: float poissons ratio
    :param R: float indenter radius
    :param Gg: float instantaneous shear modulus
    :param Ge: float equilibrium shear modulus
    :param Tau: float relaxation time
    :param t: float time axis
    :return: float force
    '''
    dt = t[1] - t[0]
    G = Gg - Ge
    if G < 0:
        print('invalid moduli')
    exp = np.exp(-t / Tau)
    a = 4 * np.sqrt(R) / (3 * (1 - v ** 2))
    H = inden ** 1.5
    return a * (Gg * H - G / Tau * np.convolve(exp, H, 'full')[: t.size] * dt)

def ext_lee_radok_hc(z, z0, Gg, Ge, Tau, v, R, t):
    '''
    calculate the extended viscoelastic (lee-radok) contact force, for the hard-contact case, as a function of
    the indenter position and sample equilibrium position
    :param z: float indenter position
    :param z0: float sample equilibrium position
    :param Gg: float instantaneous shear modulus
    :param Ge: float equilibrium shear modulus
    :param Tau: float relaxation time
    :param v: float poissons ratio
    :param R: float indenter radius
    :param t: float time axis
    :return: float force
    '''
    z_shift = z - z0
    chi = (abs(z_shift) - z_shift) / 2
    return lee_radok(chi, v, R, Gg, Ge, Tau, t)

def get_r(x):
    '''
    get the time decay constant needed for a given signal
    :param x: some signal (listlike)
    :return: optimal time decay constant (float)
    '''
    first_nonzero = np.nonzero(x)[0][0]
    return abs(x[-1] / x[first_nonzero]) ** (1 / (x.size - first_nonzero))


def mdft(x, r, length=None):
    '''
    perform the modified discrete fourier transform of a signal at a given radial distance
    :param x: some signal (listlike)
    :param r: radial distance defining the circle on which the modified fourier transform will be calculated (float)
    (r=1.0 gives a discrete fourier transform)
    :param length: number of elements in the transformed signal (int)
    (default is None, which leaves length=len(x); recommended to set equal to the length of shortest signal in batch)
    :return: modified discrete fourier transform of x at r (numpy array)
    '''
    n = np.arange(0, x.size, 1)
    return np.fft.fftshift(np.fft.fft(x * r ** -n, n=length))


def get_avg_q(stresses, strains, r_mult=1):
    N = min(x.size for x in stresses)
    r = max(max(get_r(f_s), get_r(f_e)) for f_s, f_e in zip(stresses, strains)) * r_mult
    q = np.mean([mdft(f_s, r, N) for f_s in stresses], axis=0) / np.mean([mdft(f_e, r, N) for f_e in strains], axis=0)
    return q

def qmax(X, r, length, dt):
    '''
    relaxance (Q) of a single arm maxwell (standard linear solid) model in the Z-domain
    :param X: model parameters (array of three floats) [Ge, G1, T1] where Ge is the equilibrium modulus in units of Pa,
    G1 is the modulus of the maxwell arm in units of Pa, and T1 is the relaxation time in units of s^-1. The sum of
    Ge and G1 is known as Gg, the glassy modulus or instantaneous response (analogous to young's modulus)
    :param r: radial distance defining the circle on which the modified fourier transform will be calculated (float)
    (r=1.0 gives a discrete fourier transform)
    :param freq: frequency array in Hz (listlike) [-Nyquist freq, Nyquist freq]
    :return: relaxance of a single arm maxwell model in the Z-domain along the circle of radius r (numpy array)
    '''
    Ge, G1, T1 = X
    z = np.exp(np.pi * 1j * np.linspace(-1, 1, length)) * r
    return (Ge + G1) - G1 / (1 + T1 / dt * (1 - 1 / z))

def qmax_obj(X, real, r, dt, re_weight=1, im_weight=1):
    pred = qmax(X, r, real.size, dt)
    pred = abs(np.real(pred)) + abs(np.imag(pred)) * 1j
    return np.sum(re_weight * abs(np.real(pred - real)) ** 2 + im_weight * abs(np.imag(pred - real)) ** 2)


def q_sls(X, s):
    Ge = X[0]
    G = X[1::2]
    T = X[2::2]
    return Ge + np.sum(G) - np.sum([g / (1 + t * s) for g, t in zip(G, T)], axis=0)