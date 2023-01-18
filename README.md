Strong lensing modeling of the six Hubble Frontier Fields clusters 
==================================================================

Code files for evaluating multiple image scatters
--------------------------------------------
A code file named 'eval_source_scatter.py' generates the source plane scatters for multiple images.

In a folder named 'Image_plane_scatters', there is the code for computing image plane scatters for multiple images.
A 'lens_rms.py' is the code to find the location of the multiple images in the image plane.
A 'plot_result.py' is the code to plot the result rms scatters.
We upload our results and code for computing scatters in both the source and the image planes.

Result files from the MARS algorithm
--------------------------------------------
In each folder, there are 'result_fits.fits', 'resut_kappa_w_header.fits', 'deflection_angle_w_header.fits', and 'catalog.txt' files. 
All kappa and deflection angle maps are scaled to D<sub>ds</sub>/D<sub>s</sub>= 1.

A 'result_fits.fits' file is our result SL model which obtains convergence and model redshift values.
'result_fits.fits' file has a size of (140x140 + alpha), where alpha is the number of model redshifts.
A 'result_kappa_w_header.fits' file is our result SL kappa map (100x100) with WCS header. 
A 'deflection_angle_w_header.fits' file is our result deflection angle map (100x100) with WCS header.
A 'catalog.txt' file is the catalog file for SL reconstruction in pixel scale.


For more details, please refer the ''.
