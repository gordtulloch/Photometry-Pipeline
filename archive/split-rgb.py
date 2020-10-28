#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
=====================================================
Convert a 3-color image (FITS) to separate FITS images
=====================================================

This example opens an RGB FITS image and writes out each channel as a separate
FITS (image) file, copying all the headers and setting COLORSPC to the correct value

*By: Gord Tulloch from code by Erik Bray, Adrian Price-Whelan*

*License: BSD*


"""

import numpy as np
from PIL import Image
from astropy.io import fits
import sys
import getopt

inputfile = sys.argv[1]
##############################################################################
# Load and display the original 3-color fits image:
hdu_list = fits.open(inputfile)
hdu_list.info()

##############################################################################
# Extract the image data and split into R G B channels
image_data = hdu_list[0].data
indices=(0, 1, 2)
image_r = image_data[indices[0], :, :]
image_g = image_data[indices[1], :, :]
image_b = image_data[indices[2], :, :]

##############################################################################
# Init the rgb images with the correct image header
image_header = hdu_list[0].header
print(image_header)
red = fits.PrimaryHDU(data=image_r)
red.header=hdu_list[0].header
red.header.set('COLORSPC', 'R       ', 'PCL: Color space')
red.writeto('red.fits')

green = fits.PrimaryHDU(data=image_g)
green.header=hdu_list[0].header
green.header.set('COLORSPC', 'G       ', 'PCL: Color space')
green.writeto('green.fits')

blue = fits.PrimaryHDU(data=image_b)
blue.header=hdu_list[0].header
blue.header.set('COLORSPC', 'B       ', 'PCL: Color space')
blue.writeto('blue.fits')

hdu_list.close()
##############################################################################
# Write out the channels as separate FITS images



