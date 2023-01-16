import numpy as np
from Function_torch import cos_z
import torch
from load_parameter_torch import strong_lensing_file, cluster_z, x1, y1, x2, y2, margin,kapmap_size, device

kappamap_size = kapmap_size


#load data from strong_lensing_data_file
strong_lensing_data = np.loadtxt(strong_lensing_file, usecols=(1, 2, 3, 4, 5))
strong_image_index = np.loadtxt(strong_lensing_file, usecols=0, dtype=str)
strong_lensing_data = strong_lensing_data.T

strong_lensing_data_count = len(strong_lensing_data[0])
strong_source_number = torch.tensor(strong_lensing_data[0], dtype=torch.float32, device=device)
strong_image_x = torch.tensor(strong_lensing_data[1], dtype=torch.float32, device=device)
strong_image_y = torch.tensor(strong_lensing_data[2], dtype=torch.float32, device=device)
strong_source_redshift = torch.tensor(strong_lensing_data[3], dtype=torch.float32, device=device)
strong_source_image_weight = torch.tensor(strong_lensing_data[4], dtype=torch.float32, device=device)


#convert source index to source number
source_number_image = []
for number in range(len(strong_image_index)):
    source_number_image.append(float(strong_image_index[number][1:]))
source_number_image = torch.tensor(np.array(source_number_image, dtype=int), dtype=torch.int32, device=device)


# load strong-lensing data
galaxy_m_w = torch.zeros((len(strong_source_redshift)), dtype=torch.float32, device=device)
for redshift, i in zip(strong_source_redshift, range(len(strong_source_redshift))):
    galaxy_m_w[i] = cos_z(redshift, torch.tensor(cluster_z, device=device))
galaxy_m_x = ((strong_image_x - x1) / (x2 - x1) * (kappamap_size - 1) + margin).clone().detach().to(device)
galaxy_m_y = ((strong_image_y - y1) / (y2 - y1) * (kappamap_size - 1) + margin).clone().detach().to(device)


# calculate the total source galaxy number & knot
max_m_source = int(torch.max(strong_source_number)) + 1
