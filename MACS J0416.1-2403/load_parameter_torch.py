bg_galaxy_file = None
strong_lensing_file = 'MACSJ0416_catalog.txt'
interpol_size = 2500
cluster_z = 0.396
x1 = 0
y1 = 0
x2 = x1 + interpol_size
y2 = y1 + interpol_size
kapmap_size = 100
margin = 20
padding = int(2*margin + kapmap_size)
upper_red = 15
free_param_redshift_source = None#8, 11, 41, 43, 44, 45

import torch
device = torch.device('cpu')
#device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
