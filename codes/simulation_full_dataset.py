from gelsa import visu
from profile_fitter import Profile_Fitter
from spectra_generator import Spectra_Generator
import numpy as np
from tqdm.notebook import tqdm
from gelsa import Gelsa
from gelsa import analysis
from gelsa import galaxy
from gelsa import visu
from astropy.io import fits
from astropy.table import Table, unique
from astropy.constants import c
import astropy.units as u
import astropy.coordinates as coord
from astroquery.vizier import Vizier
import os
import argparse

# ----------------------------------------------- Parse command-line arguments ---------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("pt_id", type=int, help="Pointing ID (e.g. 30477)")
parser.add_argument("--grism", type=str, default="RGS000_0", help="Grism + tilt combination")
parser.add_argument("--lines", action="store_true", help="Simulate emission lines or not")
args = parser.parse_args()
# ---------------------------------------------------- Load EDFN catalog ---------------------------------------------------
filename = "/scratch/astro/nicolo.fiaba/EDFN_deep.fits"
hdu_list = fits.open(filename, memmap=True)
data = Table(hdu_list[1].data)

"""
Pointing ID (e.g. 30477)
"""
pt_id = args.pt_id

# --------------------------------------------------------------------------------------------------------------------------
pointing_filename = "/scratch/astro/nicolo.fiaba/pointing_list.txt"
pointings = Table.read(pointing_filename, format='ascii')

# One ID number refers to two rows in the pointings file (usually), so we remove duplicates
pointings = unique(pointings, keys='Data.ObservationSequence.PointingId')

pt_id_num = np.where(pointings["Data.ObservationSequence.PointingId"] == pt_id)[0][0]

if pt_id in pointings["Data.ObservationSequence.PointingId"]:
    print(f"\n \t Simulating pointing ID: {pt_id}\n")
else: 
    print("ID not valid!")

pt_ra = pointings["Data.AdjustedPointing.RA"][pt_id_num]
pt_dec = pointings["Data.AdjustedPointing.Dec"][pt_id_num]

# --------------------------------------------------- Load 2MASS catalog ----------------------------------------------------
vizier = Vizier()
vizier.ROW_LIMIT = 1e6
result = vizier.query_region(coord.SkyCoord(ra=pt_ra, dec=pt_dec,
                                            unit=(u.deg, u.deg),
                                            frame='icrs'),
                        width='0.763deg',
                        height='0.722deg',
                        catalog="II/246/out")

data_2mass = result[0]

# Masking out the VERY bright sources
jmag_threshold = 11.5
data_2mass_th = data_2mass[data_2mass['Jmag'] > jmag_threshold]
#----------------------------------------------------------------------------------------------------------------------------

smj_centers = [5, 15, 20, 25, 30, 35, 45, 50]

# Coefficients for pt_id 30477: We take these as the coefficients for ALL POINTINGS!
if 'pt1_coeffs' not in globals():
    pt1_coeffs = [[ 1.77234446e-03,  2.53673518e+00],
        [ 4.07338437e-03,  1.87170864e+00],
        [ 1.12526831e-02,  2.08003500e+00],
        [ 2.32253015e-02,  1.71388283e+00],
        [ 1.17772560e-02,  1.50861189e+00],
        [ 2.39869540e-03,  1.78675470e+00],
        [-3.13389171e-02,  2.95207141e+00],
        [-5.49549843e-02,  3.87116259e+00],
        [ 5.49966246e-03,  1.92784429e+00],
        [ 1.39497894e-02,  1.96737090e+00],
        [ 1.04667474e-02,  1.62507997e+00],
        [ 1.06938595e-02,  1.35771835e+00],
        [ 2.33678708e-02,  1.22726095e+00],
        [-3.74385235e-02,  3.39684561e+00],
        [ 8.18984806e-03,  2.04066593e+00],
        [-1.00264718e-03,  2.30374499e+00]]

pt1_coeffs_2mass = [[-0.03374102,  2.68011832],
       [ 0.06543262,  1.18695557],
       [ 0.05265686,  1.32283175],
       [ 0.0327894 ,  1.60266054],
       [ 0.03782269,  1.534863  ],
       [ 0.03715364,  1.50875354],
       [ 0.17387974, -0.43654132],
       [ 0.04852676,  1.31908536],
       [ 0.02510635,  1.65687156],
       [ 0.05222355,  1.30844128],
       [ 0.00960759,  1.97192287],
       [ 0.054146  ,  1.21343136],
       [ 0.05096768,  1.29499245],
       [ 0.11594915,  0.37683558],
       [ 0.00326621,  2.09770846],
       [ 0.08651997,  0.83806574]]

# Initializing Gelsa
G = Gelsa(config_file="/scratch/astro/nicolo.fiaba/gelsa-spectra/calib/gelsa_config.json", 
                        calibdir="/scratch/astro/nicolo.fiaba/gelsa-spectra/calib/", zero_order_catalog=None)

# --------------------------------------------------- Fluxes for EDFN galaxies ----------------------------------------------------
# We compute the line between two flux values in J and H band

c_cgs = c.cgs.value
# I compute the conversion factor c/lambda^2
factor = lambda lam: (c_cgs*1e8)/(lam**2)
# Center wavelength values in angstrom
j_center =  13673
h_center = 17714
continuum_wavelength = np.linspace(11999, 19001, 100)

# first, I convert the fluxes to erg s^-1 cm^-2 Hz^-1
flux_y_nu = 1e-29*data["FLUX_Y_TEMPLFIT_MWCORR"]
flux_j_nu = 1e-29*data["FLUX_J_TEMPLFIT_MWCORR"]
flux_h_nu = 1e-29*data["FLUX_H_TEMPLFIT_MWCORR"]

flux_j_lam = flux_j_nu * factor(j_center)
flux_h_lam = flux_h_nu * factor(h_center)

log_j_flux = np.log10(flux_j_lam)
log_h_flux = np.log10(flux_h_lam)

axis_ratio = 1 - data['ELLIPTICITY']
data["AXIS_RATIO"] = axis_ratio

x1, x2 = j_center, h_center
y1, y2 = np.array(log_j_flux), np.array(log_h_flux)

m = (y2 - y1) / (x2 - x1)
b = y1 - m * x1

# Now compute y-values (fluxes) for all continuum wavelengths
fluxes = m[:, None] * continuum_wavelength[None, :] + b[:, None]
fluxes = np.array(fluxes)

# --------------------------------------------------- Fluxes for 2MASS stars ----------------------------------------------------
Jmag = data_2mass_th['Jmag']
Hmag = data_2mass_th['Hmag']
Kmag = data_2mass_th['Kmag']

J_0 = 1594
H_0 = 1024
K_0 = 666.7

mag_to_flux = lambda m, f_0: f_0 * 10**(- m / 2.5)

flux_J = mag_to_flux(Jmag, J_0) * 1e-23 #specific flux in Jy * 10^-23 = specific flux in cgs
flux_H = mag_to_flux(Hmag, H_0) * 1e-23
flux_K = mag_to_flux(Kmag, K_0) * 1e-23

flux_j_lam_2mass = flux_J * factor(j_center)
flux_h_lam_2mass = flux_H * factor(h_center)

log_j_flux_2mass = np.log10(flux_j_lam_2mass)
log_h_flux_2mass = np.log10(flux_h_lam_2mass)

y1_2mass, y2_2mass = np.array(log_j_flux_2mass), np.array(log_h_flux_2mass)

m_2mass = (y2_2mass - y1_2mass) / (x2 - x1)
b_2mass = y1_2mass - m_2mass * x1

# Now compute y-values (fluxes) for all continuum wavelengths
twomass_fluxes = m_2mass[:, None] * continuum_wavelength[None, :] + b_2mass[:, None]
twomass_fluxes = np.array(twomass_fluxes)

#--------------------------------------------------------------------------------------------------------------------------------
"""
Specify the detector number you want to simulate.
If no detector number is provided, all detectors will be simulated in parallel. 
In that case make sure to allocate 16 CPUs on the cluster.
""";

SG = Spectra_Generator(pointings = pointings, 
                       data = data, 
                       twomass_data = data_2mass_th,
                       continuum_wavelength = continuum_wavelength, 
                       pt_n = pt_id_num, 
                       pt_id = pt_id,
                       fluxes = fluxes, 
                       twomass_fluxes = twomass_fluxes,
                       fit_coeffs = pt1_coeffs,
                       fit_coeffs_2mass = pt1_coeffs_2mass)

frames = SG.generate_sources(lines=args.lines)
frames = np.array(frames)

#----------------------------------------------------- Export simulations to file -----------------------------------------------------------


folder = "/scratch/astro/nicolo.fiaba/simulated_images/" + args.grism

if not os.path.exists(folder):
    os.makedirs(folder)
    
SG.export_images_to_file(frames, pt_id = pt_id, folder=folder, lines=args.lines)

