import numpy as np
import scipy.ndimage as ndimage
from astropy.cosmology import FlatLambdaCDM, LambdaCDM
from scipy.fft import fft2, ifft2
from tqdm import tqdm

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
def cos_z_1(source, lensing):
    global cosmo
    Dls = cosmo.angular_diameter_distance_z1z2(lensing,source)
    Ds = cosmo.angular_diameter_distance(source)
    return (1/(Ds/Dls)).value


def reg_kappa_cross(kappa, prior):
    penalty = 1.e6 * np.max((0., 0.05 - kappa.min())) ** 2
    prior[prior < 1e-8] = 1e-8
    kappa[kappa < 1e-8] = 1e-8
    return np.sum(prior - kappa + kappa * np.log(kappa / prior)) + penalty


def convert_potential(potential):

    potential_map_size = len(potential[0])

    l = np.roll(potential, shift=1)
    r = np.roll(potential, shift=-1)
    u = np.roll(potential, shift=-1, axis=0)
    d = np.roll(potential, shift=1, axis=0)

    ul = np.roll(np.roll(potential, shift=1, axis=0), shift=1, axis=1)
    ur = np.roll(np.roll(potential, shift=1, axis=0), shift=-1, axis=1)
    dl = np.roll(np.roll(potential, shift=-1, axis=0), shift=1, axis=1)
    dr = np.roll(np.roll(potential, shift=-1, axis=0), shift=-1, axis=1)

    kappa = 0.5 * (r + l + u + d - 4 * potential)
    alpha_x = 0.5 * (r - l)
    alpha_y = 0.5 * (u - d)
    gamma1 = 0.5 * (l + r - u - d)
    gamma2 = 0.25 * (ur - ul - dr + dl)

    kappa_grid = kappa[1:potential_map_size - 1, 1:potential_map_size - 1]
    gamma1_grid = gamma1[1:potential_map_size - 1, 1:potential_map_size - 1]
    gamma2_grid = gamma2[1:potential_map_size - 1, 1:potential_map_size - 1]
    alpha_x_grid = alpha_x[1:potential_map_size - 1, 1:potential_map_size - 1]
    alpha_y_grid = alpha_y[1:potential_map_size - 1, 1:potential_map_size - 1]

    return kappa_grid, alpha_x_grid, alpha_y_grid, gamma1_grid, gamma2_grid


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


def kernel_def_pot(padding, mapsize):
    total_with_padding = mapsize + 2*padding
    x1 = np.linspace(0,total_with_padding-1,total_with_padding, dtype=np.float32)
    x2 = np.linspace(0,total_with_padding-1,total_with_padding, dtype=np.float32)
    xs1,xs2 = np.meshgrid(x1,x2)
    xs1 = xs1 - (total_with_padding/2)
    xs2 = xs2 - (total_with_padding/2)
    D = np.log(np.sqrt(xs1**2 + xs2**2))
    D[np.isinf(D)] = -1

    return D


def gamma_kernel(padding, mapsize):
    total_with_padding = mapsize + 2*padding
    x1 = np.linspace(0,total_with_padding-1,total_with_padding, dtype=np.float32)
    x2 = np.linspace(0,total_with_padding-1,total_with_padding, dtype=np.float32)
    xs1,xs2 = np.meshgrid(x1,x2)
    xs1 = xs1 - (total_with_padding/2)
    xs2 = xs2 - (total_with_padding/2)
    D = (xs2**2 - xs1**2 - 2j*xs1*xs2) / ((xs1**2 + xs2**2)**2)
    D = np.nan_to_num(D)

    return D


def conv_def_angle(kappamap, padding, Kernel):
    kappamap_size = len(kappamap[0])
    result_pad = extended_maps_zero(kappamap, padding, kappamap_size)
    deflection_angle = convolution(Kernel, result_pad, padding, kappamap_size)

    return deflection_angle[0], deflection_angle[1]


def calc_beta(alpha_x_grid, alpha_y_grid, galaxy_m_x, galaxy_m_y, galaxy_m_w):
    l_alpha_x = ndimage.map_coordinates(alpha_x_grid, [galaxy_m_y, galaxy_m_x])
    l_alpha_y = ndimage.map_coordinates(alpha_y_grid, [galaxy_m_y, galaxy_m_x])

    beta1 = galaxy_m_x - l_alpha_x * galaxy_m_w
    beta2 = galaxy_m_y - l_alpha_y * galaxy_m_w

    return beta1, beta2


def conv_gamma(kappamap, padding, Kernel):
    kappamap_size = len(kappamap[0])
    input_kappa_map_pad = extended_maps_zero(kappamap, padding, kappamap_size)
    gamma = convolution(Kernel, input_kappa_map_pad, padding, kappamap_size)

    return gamma[0], gamma[1]


def conv_mag(kappamap, padding, log=True):
    gamma = conv_gamma(kappamap, padding)
    gamma_x, gamma_y = gamma[0], gamma[1]
    magnification = 1 / ((1 - kappamap) ** 2 - (gamma_x ** 2 + gamma_y ** 2))
    mag = np.abs(magnification)

    if log == True:
        return np.log(mag)
    else:
        return mag


def src_to_img(mag, kappa, gamma1, gamma2, x1, y1):
    x = np.abs(mag * ((1-kappa+gamma1)*x1 + gamma2*y1))
    y = np.abs(mag * (gamma2*x1 + (1-kappa-gamma1)*y1))
    return x, y


def img_to_src(kappa, gamma1, gamma2, x1, y1):
    x = np.abs(((1-kappa-gamma1)*x1 - gamma2*y1))
    y = np.abs((-gamma2*x1 + (1-kappa+gamma1)*y1))
    return x, y


def conv_total(kappa, gamma1, gamma2, mag, galaxy_x, galaxy_y):
    l_kappa = ndimage.map_coordinates(kappa, [galaxy_y, galaxy_x])
    l_gamma_1 = ndimage.map_coordinates(gamma1, [galaxy_y, galaxy_x])
    l_gamma_2 = ndimage.map_coordinates(gamma2, [galaxy_y, galaxy_x])
    l_mag = ndimage.map_coordinates(mag, [galaxy_y, galaxy_x], order=1)
    return l_kappa, l_gamma_1, l_gamma_2, np.abs(l_mag)


def extended_maps_zero(original_map, padding, mapsize):
    total_with_padding = mapsize + 2*padding
    k = np.zeros((total_with_padding, total_with_padding), dtype=np.float64)
    k[padding : mapsize + padding, padding : mapsize + padding] = original_map
    return k


def find_max(data_set):
    mapSize = len(data_set[0])
    map_1d = data_set.reshape(mapSize*mapSize)
    max_value = np.argmax(map_1d)
    Central_coor = [max_value % mapSize, max_value // mapSize]
    return Central_coor


def Radius(data_set, data_unc, fin, start, r_scale, process=None):
    Max_coor = find_max(data_set)
    data_size = len(data_set[0])

    max_x, max_y = Max_coor[0], Max_coor[1]
    xarr, yarr = np.linspace(0, data_size - 1, data_size), np.linspace(0, data_size - 1, data_size)
    xarr -= max_x
    yarr -= max_y

    new_x_axis, new_y_axis = np.meshgrid(xarr, yarr)
    r = np.sqrt(new_x_axis ** 2 + new_y_axis ** 2)

    radi = np.linspace(start, fin, fin-start+1) * r_scale

    count = fin - start + 1
    kappa_rad_avg = np.zeros((count))
    kappa_rad_sig = np.zeros((count))
    kappa_rad_cumul = np.zeros((count))
    cumulative = 0

    if process == True:
        for step in tqdm(range(start, fin + 1)):
            a = np.where((step - 1 < r) & (r < step + 1))
            kappa_rad_avg[step - start] = np.mean(data_set[a])
            kappa_rad_sig[step - start] = np.mean(data_unc[a])
            cumulative += np.sum(data_set[a])
            kappa_rad_cumul[step - start] = cumulative

    else:
        for step in range(start, fin + 1):
            a = np.where((step - 1 < r) & (r < step + 1))
            kappa_rad_avg[step - start] = np.mean(data_set[a])
            kappa_rad_sig[step - start] = np.mean(data_unc[a])
            cumulative += np.sum(data_set[a])
            kappa_rad_cumul[step - start] = cumulative

    return radi, kappa_rad_avg, kappa_rad_cumul, kappa_rad_sig


def interpol_map(data_set, factor): # for 2D image
    origin_size = len(data_set[0])
    interpol_size = len(data_set[0]) * factor
    x, y = np.linspace(0, origin_size - 1, interpol_size), np.linspace(0, origin_size - 1, interpol_size)
    x_axis, y_axis = np.meshgrid(x, y)
    kappa_interpol = ndimage.map_coordinates(data_set, [y_axis, x_axis])
    return kappa_interpol


def update_redshift(update_red, galaxy_m_w, cluster_z, free_param_redshift_source, source_number_image):
    free_param_redshift_source = np.array(free_param_redshift_source)
    new_galaxy_m_w = np.copy(galaxy_m_w)
    for number2 in zip(free_param_redshift_source, range(len(free_param_redshift_source))):
        indexing = np.where(source_number_image == number2[0])
        new_redshift = update_red[number2[1]]
        redshift = np.ones((len(indexing))) * new_redshift
        m_w = cos_z_1(redshift, cluster_z)

        new_galaxy_m_w[indexing] = m_w

    return new_galaxy_m_w
