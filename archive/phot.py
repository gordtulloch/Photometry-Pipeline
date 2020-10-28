import requests, math, glob
import pandas as pd
import numpy as np
from photutils import DAOStarFinder
from astropy.stats import mad_std
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
import matplotlib.pyplot as plt
from photutils import aperture_photometry, CircularAperture
from astroquery.simbad import Simbad
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

FITS_FILE = '/home/dokeeffe/Pictures/CalibratedLight/2017-10-02/KIC08462852-2017-10-02-20-56-22Light_005.fits'
STAR_NAME = 'KIC08462852'
BRIGHTEST_COMPARISON_STAR_MAG = 11.0
DIMMEST_COMPARISON_STAR_MAG = 13.0

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

astroquery_results = Simbad.query_object(STAR_NAME)
TARGET_RA = str(astroquery_results[0]['RA'])
TARGET_DEC = str(astroquery_results[0]['DEC']).replace('+','').replace('-','')
results, chart_id = get_comp_stars(TARGET_RA, TARGET_DEC)
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
print('Magnitude estimate: {} error from residuals {}'.format(target_magnitude, residuals))
x = np.append(x,target_instrumental_magnitude)
y = np.append(y,target_magnitude)


check_star_instrumental_magnitude = results[results.auid=='000-BML-045']['instrumental_mag'].values[0]
check_magnitude = fit_fn(check_star_instrumental_magnitude)
print('Check star 000-BML-045 magnitude = {}'.format(check_magnitude))

observation_date = datetime.strptime(header['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f')
print('Star Identifier: {}'.format(STAR_NAME))
print('Date-time : {}'.format(observation_date.strftime('%Y/%m/%d/%H/%M/%S')))
print('Magnitude: {} Error: {}'.format(target_magnitude, residuals))
print('Check Star 000-BML-045 Magnitude: {}'.format(check_magnitude))
print('Chart ID: {}'.format(chart_id))
print('Ensemble of {}. Error as residuals of linear fit'.format(to_linear_fit['auid'].values))

