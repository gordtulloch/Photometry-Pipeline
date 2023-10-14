#!/usr/bin/python3
# -*- coding: utf-8 -*-

import requests, math, glob
import pandas as pd
import numpy as np
from photutils.detection import DAOStarFinder
from photutils.aperture import aperture_photometry
from photutils.aperture import CircularAperture
from astropy.stats import mad_std
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
import matplotlib.pyplot as plt
from astroquery.simbad import Simbad
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def get_comp_stars(ra,dec,filter_band='V',field_of_view=18.5):
    result = []
    vsp_template = 'https://www.aavso.org/apps/vsp/api/chart/?format=json&fov={}&maglimit=18.5&ra={}&dec={}'
    print(vsp_template.format(field_of_view, ra, dec))
    r = requests.get(vsp_template.format(field_of_view, ra, dec))
    chart_id = r.json()['chartid']
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

def process_fits(FITS_FILE,STAR_NAME,BRIGHTEST_COMPARISON_STAR_MAG,DIMMEST_COMPARISON_STAR_MAG):
    
    # Download comparison stars and search simbad for our target.
    astroquery_results = Simbad.query_object(STAR_NAME)
    TARGET_RA = str(astroquery_results[0]['RA'])
    TARGET_DEC = str(astroquery_results[0]['DEC']).replace('+','').replace('-','')
    results, chart_id = get_comp_stars(TARGET_RA, TARGET_DEC)
    print('{} comp stars found'.format(len(results)))
    results.append({'auid': 'target', 'ra': TARGET_RA, 'dec': TARGET_DEC})
    print(results)
    
    # extract sources from image and add details to comp_stars - these should be in a config file
    fwhm = 3.0
    source_snr = 20
    hdulist = fits.open(FITS_FILE)
    data = hdulist[0].data.astype(float)
    header = hdulist[0].header
    wcs = WCS(header)
    bkg_sigma = mad_std(data)    
    daofind = DAOStarFinder(fwhm=fwhm, threshold=source_snr*bkg_sigma)    
    sources = daofind(data)
    print("Sources found:")
    print(sources)
    
    # Find the sources that correspond to our target and comparison stars
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
    print("Sources corresponding to target and comp:")
    print(results)

    # Perform aperture photometry and add to the results
    results = results.query('x > 0 and y > 0')
    aperture_radius = 6.0
    positions= np.column_stack((results['x'], results['y']))  
    print("Positions to be passed:")
    print(positions)
    apertures = CircularAperture(positions, r=aperture_radius)    
    phot_table = aperture_photometry(data, apertures)     
    results['aperture_sum'] = phot_table['aperture_sum']
    # add a col with calculation for instrumental mag
    #   instrumental_mag is calculated as -2.5 * LOG10(`aperture_sum`)
    results['instrumental_mag'] = results.apply(lambda x: -2.5 * math.log10(x['aperture_sum']), axis = 1)
    print("Sources with photometry and instrumental mag:")
    print(results)

    # now perform ensemble photometry by linear regression of the comparison stars' instrumental mags
    to_linear_fit = results.query('auid != "target" and vmag > {} and vmag < {}'.format(BRIGHTEST_COMPARISON_STAR_MAG, DIMMEST_COMPARISON_STAR_MAG))
    x = to_linear_fit['instrumental_mag'].values
    y = to_linear_fit['vmag'].values
    fit, residuals, rank, singular_values, rcond = np.polyfit(x, y, 1, full=True)
    fit_fn = np.poly1d(fit)

    # fit_fn from above is a function which takes in x and returns an estimate for y, lets feed in the 'target' instrumental mag
    target_instrumental_magnitude = results[results.auid=='target']['instrumental_mag'].values[0]
    target_magnitude = fit_fn(target_instrumental_magnitude)

    # The check star should be in a database with details about the target
    check_star_instrumental_magnitude = results[results.auid=='000-BBH-921']['instrumental_mag'].values[0]
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
  
# *************************** MAINLINE *********************************

# Constants should go into a config file
FITS_FOLDER = 'data/'
FITS_FILE = '/home/gtulloch/Projects/Photometry-Pipeline/data/BGO/RWAUR-ID12496-OC145882-GR5353-I.fit'
DEBUG = 1

# These will be a database lookup based on target data in FITS file
STAR_NAME = 'RW AUR'
BRIGHTEST_COMPARISON_STAR_MAG = 8.0
DIMMEST_COMPARISON_STAR_MAG = 13.0

# Test main routine
process_fits(FITS_FILE,STAR_NAME,BRIGHTEST_COMPARISON_STAR_MAG,DIMMEST_COMPARISON_STAR_MAG)


'''# Cycle through images and process
pathlist = Path(FITS_FOLDER).glob('*.zip')
for path in pathlist:
     path_in_str = str(path)
     print(path_in_str)
     process_fits(path_in_str,STAR_NAME,BRIGHTEST_COMPARISON_STAR_MAG,DIMMEST_COMPARISON_STAR_MAG,FITS_FOLDER)
     exit(0)'''

