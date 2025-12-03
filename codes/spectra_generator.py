import numpy as np
from gelsa import Gelsa
from gelsa import galaxy
from astropy.io import fits
import os
from multiprocessing import Pool

# Set random seed
np.random.seed(42)

class Spectra_Generator:
    """
        - "pointings" is the table where the pointings are described
        - "data" is the EDFN data table, after adding the AXIS_RATIO column
        - "twomass_data" is the 2MASS data table
        - "continuum wavelength" is the array of wavelengths you want in the spectra
        - "pt_n" is the pointing index in "pointings", not the ID. (18 for 30477)
        - "fluxes" is an array containing the fluxes at each wavelength for each source. You find this with a linear fit in two NISP bands.
        - "fit_coeffs" is the (16,2) matrix of coefficients for the SEMI-MAJOR AXIS vs FWHM relation
    """
    def __init__(self, pointings, data, twomass_data, continuum_wavelength, pt_n, pt_id, fluxes, twomass_fluxes, fit_coeffs, fit_coeffs_2mass):
        self.pointings = pointings
        self.data = data
        self.twomass_data = twomass_data
        self.continuum_wavelength = continuum_wavelength
        self.pt_n = pt_n
        self.pt_id = pt_id
        self.fluxes = fluxes
        self.twomass_fluxes = twomass_fluxes
        self.fit_coeffs = fit_coeffs
        self.fit_coeffs_2mass = fit_coeffs_2mass

    # here data_ can be either EDFN or 2MASS
    def find_fwhm(self, data_, det_n, twomass=False):
        
        def fit_line(x, coeffs):
            return coeffs[0]*x + coeffs[1]

        if twomass==False:
            fwhm_arcsec = fit_line(data_["SEMIMAJOR_AXIS"], coeffs=self.fit_coeffs[det_n]) * 0.1
        else:  
            fwhm_arcsec = fit_line(data_["Jmag"], coeffs=self.fit_coeffs_2mass[det_n]) * 0.1

        return fwhm_arcsec
        
    def new_frame(self, G):
        new_frame = G.new_spec_frame(ra=self.pointings["Data.AdjustedPointing.RA"][self.pt_n], 
                                 dec=self.pointings["Data.AdjustedPointing.Dec"][self.pt_n], 
                                 pa=self.pointings["Data.AdjustedPointing.PositionAngle"][self.pt_n], 
                                 grism_name=self.pointings["Data.Grism"][self.pt_n], 
                                 tilt=self.pointings["Data.GrismWheelTilt"][self.pt_n], 
                                 clear=True
                                )
        return new_frame

    def process_detector(self, frame, det_n, temp_table, temp_flux, det1, det2, det1_2m, det2_2m, em_min, em_max, lines, z_min, z_max):
        galaxies = []
        stars = []

        mask_detector = (det1 == det_n) | (det2 == det_n) # if no detector is specified, a full FOV (16 detectors) is generated
        mask_detector_2mass = (det1_2m == det_n) | (det2_2m == det_n)
        
        print("Simulating {} galaxies in detector {}\n".format(np.sum(mask_detector), det_n))
        print("\nSimulating {} stars in detector {}\n".format(np.sum(mask_detector_2mass), det_n))
        if lines:
            print("\nSimulating emission lines also")
        
        temp_table_ondet = temp_table[mask_detector]
        temp_flux_ondet = temp_flux[mask_detector]

        twomass_ondet = self.twomass_data[mask_detector_2mass]
        twomass_flux_ondet = self.twomass_fluxes[mask_detector_2mass]
        
        fwhm_ondet = self.find_fwhm(temp_table_ondet, det_n)
        fwhm_2mass_ondet = self.find_fwhm(twomass_ondet, det_n, twomass=True)

        for k in range(len(temp_table_ondet)):
            """
            Extract random redshift from z_min to z_max
            """
            z = np.random.uniform(z_min, z_max)

            """
            Simulating galaxies
            """
            gal = galaxy.Galaxy(
                ra=temp_table_ondet["RIGHT_ASCENSION"][k],
                dec=temp_table_ondet["DECLINATION"][k],
                redshift=z,
                fwhm_arcsec=fwhm_ondet[k],
                axis_ratio=temp_table_ondet["AXIS_RATIO"][k],
                continuum_params=(15000, -1e-5, -17),
                obs_wavelength_range=(12000., 19000.)
            )
            continuum_flux = 10**(temp_flux_ondet[k])
            gal.set_sed(self.continuum_wavelength/(1+z), continuum_flux)
            # Add random emission lines
            if lines:
                log_min = np.log10(em_min)
                log_max = np.log10(em_max)
                log_flux_Ha = np.random.uniform(log_min, log_max)
                flux_Ha = 10**log_flux_Ha
                gal.set_flux_Ha(flux_Ha = flux_Ha)
                
            galaxies.append(gal)

        for k in range(len(twomass_ondet)):
            """
            Simulating stars
            """
            star = galaxy.Galaxy(
                ra=twomass_ondet['RAJ2000'][k],
                dec=twomass_ondet['DEJ2000'][k],
                redshift=0,
                fwhm_arcsec=fwhm_2mass_ondet[k],
                axis_ratio=1,
                continuum_params=(15000, -1e-5, -17),
                obs_wavelength_range=(12000., 19000.)
            )
            continuum_flux_twomass = 10**(twomass_flux_ondet[k])
            star.set_sed(self.continuum_wavelength, continuum_flux_twomass)
            stars.append(star)

        all_sources = galaxies + stars
            
        frame.add_sources(all_sources, noise=False)    
   
        return frame

    def process_detector_wrapper(self, args):
        """
        Multiprocessing object Pool wants a function with a single argument, hence we need this. 
        """
        self, G, det_id, temp_table, temp_flux, det1, det2, det1_2m, det2_2m, em_min, em_max, lines, z_min, z_max = args
        frame = self.new_frame(G)
        self.process_detector(frame, det_id, temp_table, temp_flux, det1, det2, det1_2m, det2_2m, em_min, em_max, lines, z_min, z_max)
        
        return frame

    def run_parallel(self, G, temp_table, temp_flux, det1, det2, det1_2m, det2_2m, em_min, em_max, lines, z_min, z_max, det_n=None):
        if det_n is not None:
            print("Currently simulating sources in detector ", det_n)
            frame = self.new_frame(G)
            self.process_detector(frame, det_n, temp_table, temp_flux, det1, det2, det1_2m, det2_2m, em_min, em_max, lines, z_min, z_max)
            return [frame]
        else:
            print("Currently simulating sources in the entire FOV")
            d_range = range(16)
            with Pool(16) as p:
                results = p.map(self.process_detector_wrapper, 
                                [(self, G, d, temp_table, temp_flux, det1, det2, det1_2m, det2_2m, em_min, em_max, lines, z_min, z_max) for d in d_range])
            
            return results

    def generate_sources(self, det_n=None, em_min=5e-17, em_max=1e-15, z_min=0.9, z_max=1.8, lines=False):
        """
        Generate spectra for one specific detector (0–15) or the entire FOV (all 16).
        """
        
        if det_n is not None and not (0 <= det_n < 16):
            print("Detector number not valid\n")
            return

        # Initialize Gelsa 
        G = Gelsa(config_file="/scratch/astro/nicolo.fiaba/gelsa-spectra/calib/gelsa_config.json", 
                  calibdir="/scratch/astro/nicolo.fiaba/gelsa-spectra/calib/", 
                  zero_order_catalog=None
                 )
        
        frame = self.new_frame(G)

        #----------------------------------------------------- EDFN sources selection ------------------------------------------------------
        """
        Selecting galaxies (EDFN catalog)
        """
        
        # Select the galaxies within 1 degrees radius from the center of every pointing
        galaxies = []

        ra_c = self.pointings["Data.AdjustedPointing.RA"][self.pt_n]
        dec_c = self.pointings["Data.AdjustedPointing.Dec"][self.pt_n]
        
        selection_mask = np.sqrt((ra_c - self.data["RIGHT_ASCENSION"])**2 + 
                                (dec_c - self.data["DECLINATION"])**2) < 1
        
        print("{} sources in pointing number {} (1 degree from center)".format(sum(selection_mask), self.pt_n))
        
        temp_table = self.data[selection_mask]
        temp_flux = self.fluxes[selection_mask]
        
        # Check which galaxies are falling outside the detectors
        x1, y1, det1 = frame.radec_to_pixel(temp_table["RIGHT_ASCENSION"], temp_table["DECLINATION"], wavelength=12000)
        x2, y2, det2 = frame.radec_to_pixel(temp_table["RIGHT_ASCENSION"], temp_table["DECLINATION"], wavelength=19000)

        #----------------------------------------------------- 2MASS sources selection ------------------------------------------------------
        """
        Selecting pointlike sources (stars from 2MASS catalog)
            2MASS sources don't need to be selected in the way shown above
            In "simulation_full_dataset.py" we load the desired region directly!
        """
        # We just need to find det1 and det2: In which detector the left and right edges are falling?
        x1_2m, y1_2m, det1_2m = frame.radec_to_pixel(self.twomass_data['RAJ2000'], self.twomass_data['DEJ2000'], wavelength=12000)
        x2_2m, y2_2m, det2_2m = frame.radec_to_pixel(self.twomass_data['RAJ2000'], self.twomass_data['DEJ2000'], wavelength=19000)

        #------------------------------------------------- Adding all sources to the frame --------------------------------------------------
        """
        Merging the tables and fluxes for the two categories of sources and add all sources to the Gelsa frame
        """
        frames = self.run_parallel(G, temp_table, temp_flux, det1, det2, det1_2m, det2_2m, det_n=det_n, 
                                   em_min=em_min, em_max=em_max, lines=lines, z_min=z_min, z_max=z_max)
        return frames
        
    def export_images_to_file(self, frames, pt_id, det_n=None, lines=False, folder=None):
        if folder is None:
            folder = "/scratch/astro/nicolo.fiaba/simulated_images"
            
        filename = f"simulated_images_{pt_id}"
        
        if lines:
            filename += "_EL"
            
        filename += ".fits"

        output_path = f"{folder}/{filename}"
        
        hdus = [fits.PrimaryHDU()]
        hdus[0].header["NDET"] = 1 if det_n is not None else 16

        if det_n is not None:
            im, _, _ = frames[0].get_detector(det_n)
            hdu_im = fits.ImageHDU(im.astype(np.float32), name=f"DET{det_n:02d}_IM")
            hdus.append(hdu_im)
            print(f"Added detector {det_n} to fits file")
        else:
            for i, f in enumerate(frames):
                for d in f._data.keys():
                    if d == i:
                        im, _, _ = f.get_detector(d)
                        hdu_im = fits.ImageHDU(im.astype(np.float32), name=f"DET{d:02d}_IM")
                        hdus.append(hdu_im)
                        print(f"Added detector {d} to fits file")
                        
        fits.HDUList(hdus).writeto(output_path, overwrite=True)
        print(f"\n Saved images to {output_path}")
            
        