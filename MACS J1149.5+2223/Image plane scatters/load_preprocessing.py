import numpy as np
from Function import cos_z_1
from load_parameter import bg_galaxy_file, strong_lensing_file, cluster_z, x1, y1, x2, y2, \
    margin,kapmap_size

kappamap_size = kapmap_size
strong_lensing_data = np.loadtxt(strong_lensing_file, usecols=(1, 2, 3, 4, 5))
strong_image_index = np.loadtxt(strong_lensing_file, usecols=0, dtype=str)
strong_lensing_data = strong_lensing_data.T

# load data from weak_lensing_data file
if not bg_galaxy_file == None:
    bg_galaxy_data = np.loadtxt(bg_galaxy_file)
    galaxy_count = bg_galaxy_data.shape[0]
    bg_galaxy_data = bg_galaxy_data.T
    galaxy_image_x = bg_galaxy_data[1]
    galaxy_image_y = bg_galaxy_data[2]
    e1 = bg_galaxy_data[3]
    e2 = bg_galaxy_data[4]
    source_redshift = bg_galaxy_data[5]
    de = bg_galaxy_data[6]
else:
    bg_galaxy_data = 0
    galaxy_count = 0
    galaxy_image_x = 0
    galaxy_image_y = 0
    e1 = 0
    e2 = 0
    source_redshift = 0
    de = 0

# calculate the shear
calculated_bg_number = 0
if not bg_galaxy_file == 'none':
    bg_galaxy_calculated = np.zeros((3, galaxy_count))
    # (w gx gy e1 e2 de)
    for i in range(galaxy_count):
        w = cos_z_1(source_redshift[i], cluster_z)
        gx = (galaxy_image_x[i] - x1) / (x2 - x1) * (kappamap_size - 1)
        gy = (galaxy_image_y[i] - y1) / (y2 - y1) * (kappamap_size - 1)
        if kappamap_size - 1 > gx >= 0.0 and kappamap_size > gy >= 0.0:
            calculated_bg_number = calculated_bg_number + 1

        bg_galaxy_calculated[0][galaxy_count] = w
        bg_galaxy_calculated[1][galaxy_count] = gx
        bg_galaxy_calculated[2][galaxy_count] = gy
else:
    bg_galaxy_calculated = 0


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
galaxy_m_w = cos_z_1(strong_source_redshift, cluster_z)
galaxy_m_x = ((strong_image_x - x1) / (x2 - x1) * (kappamap_size - 1)) + margin
galaxy_m_y = ((strong_image_y - y1) / (y2 - y1) * (kappamap_size - 1)) + margin


# calculate the total source galaxy number & knot
max_m_source = int(np.max(strong_source_number)) + 1

for i in range(max_m_source):
    a = np.where(strong_source_number == i)

