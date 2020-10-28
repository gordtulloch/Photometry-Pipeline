#!/usr/bin/python3
# -*- coding: utf-8 -*-

import requests, math, glob
import pandas as pd
import numpy as np
from photutils import DAOStarFinder
from astropy.stats import mad_std
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
from photutils import aperture_photometry, CircularAperture
from astroquery.simbad import Simbad
from datetime import datetime
import warnings
import mariadb
import sys
warnings.filterwarnings('ignore')
from pathlib import Path

def get_comp_stars(ra,dec,filter_band='V',field_of_view=18.5):
    result = []
    vsp_template = 'https://www.aavso.org/apps/vsp/api/chart/?format=json&fov={}&maglimit=18.5&ra={}&dec={}'
    if DEBUG == 1:
        print(vsp_template.format(field_of_view, ra, dec))
    r = requests.get(vsp_template.format(field_of_view, ra, dec))
    chart_id = r.json()['chartid']
    if DEBUG == 1:
        print('Downloaded Comparison Star Chart ID {}'.format(chart_id))
    for star in r.json()['photometry']:
        comparison = {}
        comparison['auid'] = star['auid']
        comparison['ra'] = star['ra']
        comparison['dec'] = star['dec']
        for band in star['bands']:
            if band['band'] == filter_band:
                comparison['vmag'] = band['mag']
                comparison['error'] = band['error']
        result.append(comparison)
    return result, chart_id

def splitrgb(FITS_FILE):
    inputfile = FITS_FILE
    hdu_list = fits.open(inputfile)

    image_data = hdu_list[0].data
    indices=(0, 1, 2)
    image_r = image_data[indices[0], :, :]
    image_g = image_data[indices[1], :, :]
    image_b = image_data[indices[2], :, :]

    image_header = hdu_list[0].header
    red = fits.PrimaryHDU(data=image_r)
    red.header=hdu_list[0].header
    red.header.set('COLORSPC', 'R       ', 'PCL: Color space')
    red.writeto('tmp/red.fits')
    green = fits.PrimaryHDU(data=image_g)
    green.header=hdu_list[0].header
    green.writeto('tmp/green.fits')
    blue = fits.PrimaryHDU(data=image_b)
    blue.header=hdu_list[0].header
    blue.header.set('COLORSPC', 'B       ', 'PCL: Color space')
    blue.writeto('tmp/blue.fits')
    hdu_list.close()
    return

def process_fits(FITS_FILE,STAR_NAME,BRIGHTEST_COMPARISON_STAR_MAG,DIMMEST_COMPARISON_STAR_MAG,FITS_FOLDER):
    astroquery_results = Simbad.query_object(STAR_NAME)
    TARGET_RA = str(astroquery_results[0]['RA'])
    TARGET_DEC = str(astroquery_results[0]['DEC']).replace('+','').replace('-','')
    results, chart_id = get_comp_stars(TARGET_RA, TARGET_DEC)
    if DEBUG == 1:
        print('{} comp stars found'.format(len(results)))
    results.append({'auid': 'target', 'ra': TARGET_RA, 'dec': TARGET_DEC})

    # extract sources from image and add details to comp_stars
    fwhm = 3.0
    source_snr = 20
    hdulist = fits.open(FITS_FILE)
    data = hdulist[0].data.astype(float)
    header = hdulist[0].header
    wcs = WCS(header)
    bkg_sigma = mad_std(data)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=source_snr*bkg_sigma)
    sources = daofind(data)

    for star in results:
        star_coord = SkyCoord(star['ra'],star['dec'], unit=(u.hourangle, u.deg))
        xy = SkyCoord.to_pixel(star_coord, wcs=wcs, origin=1)
        x = xy[0].item(0)
        y = xy[1].item(0)
        for source in sources:
            if(source['xcentroid']-4 < x < source['xcentroid']+4) and source['ycentroid']-4 < y < source['ycentroid']+4:
                star['x'] = x
                star['y'] = y
                star['peak'] = source['peak']
    results = pd.DataFrame(results)

    aperture_radius =6.0
    positions = (results['x'], results['y'])
    apertures = CircularAperture(positions, r=aperture_radius)
    phot_table = aperture_photometry(data, apertures)
    results['aperture_sum'] = phot_table['aperture_sum']
    # add a col with calculation for instrumental mag
    results['instrumental_mag'] = results.apply(lambda x: -2.5 * math.log10(x['aperture_sum']), axis = 1)

    # now perform ensemble photometry by linear regression of the comparison stars' instrumental mags
    to_linear_fit = results.query('auid != "target" and vmag > {} and vmag < {}'.format(BRIGHTEST_COMPARISON_STAR_MAG, DIMMEST_COMPARISON_STAR_MAG))
    x = to_linear_fit['instrumental_mag'].values
    y = to_linear_fit['vmag'].values
    fit, residuals, rank, singular_values, rcond = np.polyfit(x, y, 1, full=True)
    fit_fn = np.poly1d(fit)

    # fit_fn from above is a function which takes in x and returns an estimate for y, lets feed in the 'target' instrumental mag
    target_instrumental_magnitude = results[results.auid=='target']['instrumental_mag'].values[0]
    target_magnitude = fit_fn(target_instrumental_magnitude)

    check_star_instrumental_magnitude = results[results.auid=='000-BJS-730']['instrumental_mag'].values[0]
    check_magnitude = fit_fn(check_star_instrumental_magnitude)

    # Output results to file
    observation_date = datetime.strptime(header['DATE-OBS'], '%Y-%m-%dT%H:%M:%S')
    outputfilename = FITS_FOLDER+STAR_NAME.replace(" ", "") + '-' + header['DATE-OBS'].replace(":", "")+'.aavso'
    if DEBUG == 1:
        print('Filename: {}'.format(outputfilename))
    text_file = open(outputfilename, "w")
    n = text_file.write('Star Identifier: {}\n'.format(STAR_NAME))
    n = text_file.write('Date-time : {}\n'.format(observation_date.strftime('%Y/%m/%d/%H/%M/%S')))
    n = text_file.write('Magnitude: {} Error: {}\n'.format(target_magnitude, residuals))
    n = text_file.write('Check Star 000-BJS-730 Magnitude: {}\n'.format(check_magnitude))
    n = text_file.write('Chart ID: {}\n'.format(chart_id))
    n = text_file.write('Ensemble of {}. Error as residuals of linear fit\n'.format(to_linear_fit['auid'].values))
    text_file.close()
    return

# **************************************
# * Connect to MariaDB Platform
def connectDB():
    try:
        conn = mariadb.connect(
        	user="phot",
        	password="phot432ometry",
        	host="localhost",
        	port=3306,
        	database="phot"
    	)
    except mariadb.Error as e:
       print(f"Error connecting to MariaDB Platform: {e}")
       sys.exit(1)

    # Get Cursor
    cur = conn.cursor()
    return cur
    
# *************************** MAINLINE *********************************

# Set up database
cur = connectDB()

# Other constants
FITS_FOLDER = 'data/'
STAR_NAME = 'AG DRA'
BRIGHTEST_COMPARISON_STAR_MAG = 8.0
DIMMEST_COMPARISON_STAR_MAG = 13.0
DEBUG = 1

# Cycle through images and process
pathlist = Path(FITS_FOLDER).glob('*.fit')
for path in pathlist:
     path_in_str = str(path)
     print(path_in_str)
     splitrgb(path_in_str)
     process_fits(path_in_str,STAR_NAME,BRIGHTEST_COMPARISON_STAR_MAG,DIMMEST_COMPARISON_STAR_MAG,FITS_FOLDER)


