# Fast-Contact-Dimer-Spectroscopy
This repo is for all code associated with the fast contact dimer spectroscopy project.

The summary folder is a cleaned up version of the data analysis used for the paper _. To run all of this code you need to use 

General files:
To use data_class.py:
#to use fit example:
	Data("filename").fit(fit_func=One you want, names=['x','y'])
#to use multiplot ex:
	Data(“filename”).multiplot(fit_func, names=[‘x’,’y’],avg=’x’)
#to use avg data 
	Data("filename",average_by='x')
#to exclude by a certain x value 	
	Data("filename",exclude_range=#,exclude_range_x='x')

To see an ac dimer fit with the convolved lineshipe use acdimer_lineshape_fit_pulseconvolve_FDgauss.py. 

To plot Fig.2 use Plot = 1 in create_manuscript_figures_v2.py. 

