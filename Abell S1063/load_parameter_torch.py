bg_galaxy_file = None
strong_lensing_file = 'AS1063_catalog.txt'
interpol_size = 3000
cluster_z = 0.348
x1 = 0
y1 = 0
x2 = x1 + interpol_size
y2 = y1 + interpol_size
kapmap_size = 100
margin = 20
padding = int(2*margin + kapmap_size)
upper_red = 15
free_param_redshift_source = 3, 8, 9, 16, 21, 23, 31, 32, 34, 43, 45, 46, 47, 49, 50, 51, 53, 54, 55, 56

import torch
device = torch.device('cpu')
#device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
