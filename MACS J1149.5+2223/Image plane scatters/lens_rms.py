import numpy as np
from Function import conv_def_angle, kernel_def, calc_beta, update_redshift
from scipy.optimize import minimize
import scipy.ndimage as ndimage
from load_parameter import padding, kapmap_size, margin, free_param_redshift_source, cluster_z
from load_preprocessing import galaxy_m_w, galaxy_m_x, galaxy_m_y, max_m_source, strong_source_number, source_number_image
import astropy.io.fits
import parmap

kappamap_size = kapmap_size + 2*margin
result_fits = np.array(astropy.io.fits.open('./result_fits.fits')[0].data)
result_kappa = result_fits[:kappamap_size**2].reshape(kappamap_size, kappamap_size)
result_redshift = result_fits[kappamap_size**2:]
kernel_def_angle = kernel_def(padding, kapmap_size+margin*2)
alpha_x_grid, alpha_y_grid = conv_def_angle(result_kappa, padding, kernel_def_angle)

if not free_param_redshift_source == None:
    new_galaxy_m_w = update_redshift(result_redshift, galaxy_m_w, cluster_z, free_param_redshift_source, source_number_image)
else:
    new_galaxy_m_w = galaxy_m_w


beta1, beta2 = calc_beta(alpha_x_grid, alpha_y_grid, galaxy_m_x, galaxy_m_y, new_galaxy_m_w)

new_beta1, new_beta2 = np.zeros((max_m_source)), np.zeros((max_m_source))

for I in range(max_m_source):
    indexing = np.where(strong_source_number == I)
    beta1_mean, beta2_mean = np.mean(beta1[indexing]), np.mean(beta2[indexing])
    new_beta1[I], new_beta2[I] = beta1_mean, beta2_mean


def mass_min(input_coord, beta1=None, beta2=None, cosmo_weight=None):
    input_x, input_y = input_coord[0], input_coord[1]

    if input_x < margin and input_y < margin and input_x > kapmap_size+margin and input_y > kapmap_size+margin:
        return np.inf

    l_alpha_x = ndimage.map_coordinates(alpha_x_grid, [[input_y], [input_x]])
    l_alpha_y = ndimage.map_coordinates(alpha_y_grid, [[input_y], [input_x]])

    dif1 = beta1 - input_x + l_alpha_x * cosmo_weight
    dif2 = beta2 - input_y + l_alpha_y * cosmo_weight
    chi_m = (np.abs(dif1) ** 2 + np.abs(dif2) ** 2) / (1e-3)**2
    total_chi2 = np.sum(chi_m)
    return total_chi2


N =400
rng1 = np.random.RandomState(seed=12345)
input_coord_random = np.random.randint(margin, high=kapmap_size+margin, size=(N,2))


def relens(count):
    K = count
    result_values = []
    for J in range(N):
        input_coord = input_coord_random[J]
        Beta1, Beta2 = new_beta1[K], new_beta2[K]

        indexing = np.where(strong_source_number == K)
        cosmo_weight = new_galaxy_m_w[indexing][0]

        A = minimize(mass_min, input_coord, method='Powell', args=(Beta1, Beta2, cosmo_weight))
        result_chi_value, result_min_iter, result_array = A.fun, A.nit, A.x
        result_values.append(result_array.tolist())
    np.save('./lens_rms_result/multiple_image_{}.npy'.format(K), np.array(result_values))

data = range(max_m_source)
if __name__ == '__main__':
    parmap.map(relens, data, pm_pbar=True, pm_processes=40)

