# Strong lensing modeling of the six Hubble Frontier Fields clusters. 

We upload our results and code for computing scatters in both the source and the image planes.
A 'result_fits.fits' file is our result SL model which obtains convergence and model redshift values.
Each file has a size of (140x140 + alpha), where alpha is the number of model redshifts.

A code file named 'eval_source_scatter.py' generates the source plane scatters for multiple images.

In a folder named 'Image_plane_scatters', there is the code for computing image plane scatters for multiple images.
A 'lens_rms.py' is the code to find the location of the multiple images in the image plane.
A 'plot_result.py' is the code to plot the result rms scatters.
