strong_lensing_file = 'A370_catalog.txt'
interpol_size = 3000
cluster_z = 0.375
x1 = 0
y1 = 0
x2 = x1 + interpol_size
y2 = y1 + interpol_size
kapmap_size = 100
margin = 20
padding = int(2*margin + kapmap_size)
core_num = 20
free_param_redshift_source = 8, 11, 41, 43, 44, 45