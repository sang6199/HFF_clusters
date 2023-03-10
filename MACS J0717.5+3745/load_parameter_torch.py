bg_galaxy_file = None
strong_lensing_file = 'MACSJ0717_catalog.txt'
interpol_size = 4000
cluster_z = 0.545
x1 = 0
y1 = 0
x2 = x1 + interpol_size
y2 = y1 + interpol_size
kapmap_size = 100
margin = 20
padding = int(2*margin + kapmap_size)
upper_red = 15
free_param_redshift_source = 5, 8, 16, 17, 18, 20, 23, 25, 33, 34, 36, 37, 39, 45, 52, 56, 57, 65, 66, 67, 69, 76, 79, 80

import torch
device = torch.device('cpu')
#device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
