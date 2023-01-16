import numpy as np
import astropy.io.fits
import matplotlib.pyplot as plt
import torch
from Preprocessing_torch import strong_image_index, galaxy_m_w, galaxy_m_x, galaxy_m_y, \
    max_m_source, strong_source_number, source_number_image
from Function_torch import conv_def_angle, calc_beta, kernel_def, update_redshift
from load_parameter_torch import kapmap_size, interpol_size, padding, margin, free_param_redshift_source, cluster_z, device


Total_field_arcsec = 0.05 * interpol_size
arcsec_pixel = Total_field_arcsec / kapmap_size
print(arcsec_pixel)
kappamap_size = kapmap_size + 2*margin

result_fits = np.array(astropy.io.fits.open('result_fits.fits')[0].data, dtype=np.float32)
result_kappa = torch.tensor(result_fits[:kappamap_size**2].reshape(kappamap_size, kappamap_size), dtype=torch.float32, device=device)
result_redshift = torch.tensor(result_fits[kappamap_size**2:], dtype=torch.float32, device=device)

alpha_x_grid, alpha_y_grid = conv_def_angle(result_kappa , padding, kernel_def(padding, kappamap_size))

if not free_param_redshift_source == None:
    new_galaxy_m_w = update_redshift(result_redshift, galaxy_m_w, cluster_z, free_param_redshift_source, source_number_image)

else:
    new_galaxy_m_w = galaxy_m_w


beta1, beta2 = calc_beta(alpha_x_grid, alpha_y_grid, galaxy_m_x, galaxy_m_y, new_galaxy_m_w)
beta1, beta2 = beta1 * arcsec_pixel, beta2 * arcsec_pixel

beta1, beta2 = beta1.detach().cpu().numpy(), beta2.detach().cpu().numpy()

plt.figure(figsize=(5 * int((max_m_source + 1) / 6) + 1, 24))
count1, count2, RMS_total = 0, 0, 0
for P in range(max_m_source):
    a = np.where(strong_source_number == P)
    title = strong_image_index[a[0][0]]

    x_min, y_min = np.min(beta1[a]), np.min(beta2[a])
    x_max, y_max = np.max(beta1[a]), np.max(beta2[a])
    C = len(strong_source_number[a])

    beta1_mean = np.mean(beta1[a])
    beta2_mean = np.mean(beta2[a])

    count_image = len(beta1[a])
    RMS_beta = np.sqrt(np.sum(((beta1[a] - beta1_mean) ** 2) + ((beta2[a] - beta2_mean) ** 2)) / count_image)
    for_total_RMS = np.sum(((beta1[a] - beta1_mean) ** 2) + ((beta2[a] - beta2_mean) ** 2))
    RMS_total += for_total_RMS

    plt.subplot(6, int((max_m_source + 1) / 6) + 1, count2 + 1)
    plt.title(title + ': {0:5f}\'\''.format(RMS_beta))
    plt.xlim(0, x_max - x_min)
    plt.ylim(0, y_max - y_min)
    plt.scatter(beta1[a]-x_min, beta2[a]-y_min, linewidth=3.5)
    count2 = count2 + 1


print(np.sqrt(RMS_total / (max_m_source + 1)))
plt.suptitle('Total RMS: {0:5f}\'\''.format(np.sqrt(RMS_total / (max_m_source + 1))), fontsize=30)
plt.savefig('result_rms.pdf')
