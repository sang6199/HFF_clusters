import numpy as np
import scipy.ndimage as ndimage
from astropy.cosmology import FlatLambdaCDM
from scipy.fft import fft2, ifft2

#set the cosmological paramters
omegam = 0.27
vac = 0.73
omegak = 1.0-omegam-vac
H = 72
LY = 9.461e15 # m
d_H = 3000 * ((0.72)**-1) * 1e6 * (3.26 * LY)
G = 6.67408e-11 # m^3 kg^-1 s^-2
c = 299792458 # m/s
kpc = 1e3 * 3.086e16 * 100 # cm
solar = 2e33 # g
cosmo = FlatLambdaCDM(Om0=omegam, H0=H)


#using module is much faster
def cos_z(source, lensing):
    global cosmo
    Dls = cosmo.angular_diameter_distance_z1z2(lensing,source)
    Ds = cosmo.angular_diameter_distance(source)
    return (1/(Ds/Dls)).value


def convolution(kernel, kappa, padding, mapsize):
    total_with_padding = mapsize + 2*padding
    A = ifft2((fft2(kernel)*fft2(kappa))/np.pi)
    g = np.roll(A, int(total_with_padding/2), axis=0)
    g = np.roll(g, int(total_with_padding/2), axis=1)
    g = g[padding:mapsize+padding, padding:mapsize+padding]
    g1 = g.real
    g2 = g.imag
    return (g1, g2)


def kernel_def(padding, mapsize):
    total_with_padding = mapsize + 2*padding
    x1 = np.linspace(0,total_with_padding-1,total_with_padding, dtype=np.float32)
    x2 = np.linspace(0,total_with_padding-1,total_with_padding, dtype=np.float32)
    xs1,xs2 = np.meshgrid(x1,x2)
    xs1 = xs1 - (total_with_padding/2)
    xs2 = xs2 - (total_with_padding/2)
    D = (xs1 + xs2*1j)/(xs1**2 + xs2**2)
    D = np.nan_to_num(D)

    return D


def conv_def_angle(kappamap, padding, Kernel):
    kappamap_size = len(kappamap[0])
    result_pad = np.pad(kappamap, padding, 'constant', constant_values=0)
    deflection_angle = convolution(Kernel, result_pad, padding, kappamap_size)

    return deflection_angle[0], deflection_angle[1]


def calc_beta(alpha_x_grid, alpha_y_grid, galaxy_m_x, galaxy_m_y, galaxy_m_w):
    l_alpha_x = ndimage.map_coordinates(alpha_x_grid, [galaxy_m_y, galaxy_m_x])
    l_alpha_y = ndimage.map_coordinates(alpha_y_grid, [galaxy_m_y, galaxy_m_x])

    beta1 = galaxy_m_x - l_alpha_x * galaxy_m_w
    beta2 = galaxy_m_y - l_alpha_y * galaxy_m_w

    return beta1, beta2


def update_redshift(update_red, galaxy_m_w, cluster_z, free_param_redshift_source, source_number_image):
    free_param_redshift_source = np.array(free_param_redshift_source)
    new_galaxy_m_w = np.copy(galaxy_m_w)
    for number2 in zip(free_param_redshift_source, range(len(free_param_redshift_source))):
        indexing = np.where(source_number_image == number2[0])
        new_redshift = update_red[number2[1]]
        redshift = np.ones((len(indexing))) * new_redshift
        m_w = cos_z(redshift, cluster_z)

        new_galaxy_m_w[indexing] = m_w

    return new_galaxy_m_w
