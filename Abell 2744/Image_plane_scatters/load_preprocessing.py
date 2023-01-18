import numpy as np
from Function import cos_z
from load_paramter import strong_lensing_file, cluster_z, x1, y1, x2, y2, margin, kapmap_size

kappamap_size = kapmap_size
strong_lensing_data = np.loadtxt(strong_lensing_file, usecols=(1, 2, 3, 4)).T
strong_image_index = np.loadtxt(strong_lensing_file, usecols=0, dtype=str)

#load data from strong_lensing_data_file
strong_lensing_data_count = len(strong_lensing_data[0])
strong_source_number = strong_lensing_data[0]
strong_image_x = strong_lensing_data[1]
strong_image_y = strong_lensing_data[2]
strong_source_redshift = strong_lensing_data[3]
strong_source_image_weight = strong_lensing_data[4]

source_number_image = []
for number in range(len(strong_image_index)):
    source_number_image.append(float(strong_image_index[number][1:]))
source_number_image = np.array(source_number_image, dtype=int)

# load strong-lensing data
galaxy_m_w = cos_z(strong_source_redshift, cluster_z)
galaxy_m_x = ((strong_image_x - x1) / (x2 - x1) * (kappamap_size - 1)) + margin
galaxy_m_y = ((strong_image_y - y1) / (y2 - y1) * (kappamap_size - 1)) + margin


# calculate the total source galaxy number & knot
max_m_source = int(np.max(strong_source_number)) + 1

for i in range(max_m_source):
    a = np.where(strong_source_number == i)


