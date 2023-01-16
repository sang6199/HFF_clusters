import torch
from load_parameter_torch import cluster_z, upper_red, device
from torch.fft import fft2, ifft2


#set the cosmological paramters
omegam = 0.27
vac = 0.73
omegak = 1.0-omegam-vac
pi = 3.141592653589793 #from np.pi


def bicubic_interpolate_torch(im, x_arr, y_arr):
    partial_x = torch.gradient(im, dim=1)[0]  # [1]
    partial_y = torch.gradient(im, dim=0)[0]  # [0]
    partial_xy = torch.gradient(torch.gradient(im, dim=1)[0], dim=0)[0]
    number = len(x_arr.data)

    x0 = torch.floor(x_arr).long()
    x1 = x0 + 1

    y0 = torch.floor(y_arr).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    for_coef2 = torch.zeros((number, 4, 4), dtype=torch.float32, device=device)
    for_coef2[:, 0, 0] = im[y0, x0]  # f(x0, y0)
    for_coef2[:, 1, 0] = im[y0, x1]  # f(x1, y0)
    for_coef2[:, 0, 1] = im[y1, x0]  # f(x0, y1)
    for_coef2[:, 1, 1] = im[y1, x1]  # f(x1, y1)

    for_coef2[:, 2, 0] = partial_x[y0, x0]
    for_coef2[:, 3, 0] = partial_x[y0, x1]
    for_coef2[:, 2, 1] = partial_x[y1, x0]
    for_coef2[:, 3, 1] = partial_x[y1, x1]

    for_coef2[:, 0, 2] = partial_y[y0, x0]
    for_coef2[:, 1, 2] = partial_y[y0, x1]
    for_coef2[:, 0, 3] = partial_y[y1, x0]
    for_coef2[:, 1, 3] = partial_y[y1, x1]

    for_coef2[:, 2, 2] = partial_xy[y0, x0]
    for_coef2[:, 3, 2] = partial_xy[y0, x1]
    for_coef2[:, 2, 3] = partial_xy[y1, x0]
    for_coef2[:, 3, 3] = partial_xy[y1, x1]

    for_coef1 = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [-3, 3, -2, -1], [2, -2, 1, 1]], dtype=torch.float32, device=device)
    coef_mat = (for_coef1.matmul(for_coef2)).matmul(for_coef1.T)

    mat_first = torch.cat([torch.ones((number), dtype=torch.float32, device=device).reshape(number,1,1), (x_arr - x0).reshape(number,1,1), ((x_arr - x0)**2).reshape(number,1,1), ((x_arr - x0)**3).reshape(number,1,1)], dim=-1)
    mat_last = torch.cat([torch.ones((number), dtype=torch.float32, device=device).reshape(number,1,1), (y_arr - y0).reshape(number,1,1), ((y_arr - y0)**2).reshape(number,1,1), ((y_arr - y0)**3).reshape(number,1,1)], dim=1)

    result = (mat_first.matmul(coef_mat)).matmul(mat_last)
    return result.reshape(number)


# we assume Flat LCDM 
def E(z): #precisely, 1/E(z). For convenient calculation for comoving distance (dC), we use this function
    return 1/torch.sqrt(omegam*((1+z)**3) + vac)


def dC_H(z): #We don't consider H because H is vanished when we compute the cosmological weight
    z_range = z - torch.linspace(float(z),0,int(torch.round(z/1e-4)), device=device)
    E_range = E(z_range)
    return torch.trapezoid(E_range, dx=1e-4)


def cos_z_int(source, lensing): # we set reference redshift is inf (Ds_inf/Dls_inf = 1) => (Ds_inf/Dls_inf)/(Ds/Dls) = Dls/Ds
    dC_H_source = dC_H(source)
    dC_H_lensing = dC_H(lensing)
    
    Dls = (1/(1+source)) * (dC_H_source - dC_H_lensing)
    Ds = (1/(1+source)) * dC_H_source
    
    return Dls/Ds


def cos_z(source, lensing): #cubic hermite spline (https://en.wikipedia.org/wiki/Cubic_Hermite_spline)
    w = source.data
    w = w.clamp(cluster_z + 0.1 + 3e-4, upper_red - 3e-4)
    source.data = w
    
    z0 = torch.round(source * 1e4) / 1e4
    z0_1, z1, z2 = z0 - 1e-4, z0 + 1e-4, z0 + 2e-4
    f0_1, f0, f1, f2 = cos_z_int(z0_1, lensing), cos_z_int(z0, lensing), cos_z_int(z1, lensing), cos_z_int(z2, lensing)
    z_arr = torch.stack([f0_1, f0, f1, f2])
    z_grad = torch.gradient(z_arr, spacing=1e-4)
    f0_grad, f1_grad = z_grad[0][1], z_grad[0][2]    
    
    t = (source - z0) / (z1 - z0)
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h10 = t ** 3 - 2 * t ** 2 + t
    h01 = -2 * t ** 3 + 3 * t ** 2
    h11 = t ** 3 - t ** 2

    cubic_interpol = h00 * f0 + h10 * (z1 - z0) * f0_grad + h01 * f1 + h11 * (z1 - z0) * f1_grad
    return cubic_interpol


def convolution(kernel, kappa, padding, mapsize):
    total_with_padding = mapsize + 2*padding
    A = ifft2((fft2(kernel)*fft2(kappa))/3.141592653589793)  #from np.pi
    g = torch.roll(A, int(total_with_padding/2), dims=0)
    g = torch.roll(g, int(total_with_padding/2), dims=1)
    g = g[padding:mapsize+padding, padding:mapsize+padding]
    g1 = g.real
    g2 = g.imag
    return (g1, g2)


def kernel_def(padding, mapsize):
    total_with_padding = mapsize + 2*padding
    x1 = torch.linspace(0,total_with_padding-1,total_with_padding, dtype=torch.float32, device=device)
    x2 = torch.linspace(0,total_with_padding-1,total_with_padding, dtype=torch.float32, device=device)
    xs1, xs2 = torch.meshgrid(x1,x2)

    xs1, xs2 = xs1.T, xs2.T
    xs1 = xs1 - (total_with_padding/2)
    xs2 = xs2 - (total_with_padding/2)
    D = (xs1 + xs2*1j)/(xs1**2 + xs2**2)
    D1, D2 = torch.nan_to_num(D.real), torch.nan_to_num(D.imag)
    D = D1 + D2*1j

    return D


def conv_def_angle(kappamap, padding, Kernel):
    kappamap_size = len(kappamap[0])
    padding_func = torch.nn.ConstantPad2d(padding, 0)
    result_pad = padding_func(kappamap)
    deflection_angle = convolution(Kernel, result_pad, padding, kappamap_size)

    return deflection_angle[0], deflection_angle[1]


def calc_beta(alpha_x_grid, alpha_y_grid, galaxy_m_x, galaxy_m_y, galaxy_m_w):
    l_alpha_x = bicubic_interpolate_torch(alpha_x_grid, galaxy_m_x, galaxy_m_y)
    l_alpha_y = bicubic_interpolate_torch(alpha_y_grid, galaxy_m_x, galaxy_m_y)

    beta1 = galaxy_m_x - l_alpha_x * galaxy_m_w
    beta2 = galaxy_m_y - l_alpha_y * galaxy_m_w

    return beta1, beta2


def update_redshift(update_red, galaxy_m_w, cluster_z, free_param_redshift_source, source_number_image):
    new_galaxy_m_w = galaxy_m_w.data[:]

    for source_red, counting in zip(free_param_redshift_source, range(len(free_param_redshift_source))):
        indexing = torch.where(source_number_image == source_red)
        m_w = cos_z(update_red[counting], torch.tensor(cluster_z, device=device))
        new_galaxy_m_w[indexing] = m_w

    return new_galaxy_m_w


