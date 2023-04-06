Strong lensing mass reconstruction of the six Hubble Frontier Fields clusters 
==================================================================

Codes for evaluating multiple image scatters
--------------------------------------------
The code "eval_source_scatter.py" generates the source plane scatters for multiple images.

In the folder "Image_plane_scatters", there are codes for computing image plane scatters.  
"lens_rms.py": to find the location of the multiple images in the image plane.  
"plot_result.py": to plot the result rms scatters.  
We uploaded our results and code for computing scatters in both the source and the image planes.  

Results from the MARS algorithm
--------------------------------------------
In each folder, there are 'result_fits.fits', 'resut_kappa_w_header.fits', 'deflection_angle_w_header.fits', and 'catalog.txt' files.  
All kappa and deflection angle maps are scaled to D<sub>ds</sub>/D<sub>s</sub>= 1.  

"result_fits.fits" contains all parameters produced by MARS and has a size of (140x140 + alpha), where alpha is the number of model redshifts.  
"result_kappa_w_header.fits" is the 100x100 convergence map.    
"deflection_angle_w_header.fits" is the 100x100 deflection angle map in the unit of arc second.   
"catalog.txt" is the multiple image catalog. The positions are given in pixel unit.  


For more details, readers are referfed to arXiv:2301.08765. Also, feel free to contact us (sang6199@yonsei.ac.kr) if you have any questions.  
* We found errors in WCS for the fits files and re-uploaded corrected files on 2023-01-27. We thank Jori Liesenborgs for pointing this out.
* We updated files on 2023-04-06.
