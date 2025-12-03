"""
Here we're doing something similar to what's done by the profile_fitter class, but for the 2MASS sources.
Since in 2MASS we have few sources per detector, there's no need to define bins.
The x-variable is now the magnitude (Jmag for example), instead of semi-major axis.
"""

import numpy as np
import gelsa
from gelsa.sgs import datastore
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from scipy.signal import peak_widths, find_peaks

class Profile_Fitter2MASS:
    def __init__(self, cat, pt_id):
        self.cat = cat
        self.pt_id = pt_id

    def load_data_to_frame(self):
        DS = datastore.DataStore(username='', password='',
                        cachedir='/scratch/astro/benjamin.granett/datastore'
                        )
        # Load the data related to the pointing_id
        file_list = DS.load_sir_pack("DR1_R1", pointing_id_list=[self.pt_id])
        # Create a Gelsa object
        G = gelsa.Gelsa(config_file="/scratch/astro/nicolo.fiaba/gelsa-spectra/calib/gelsa_config.json", 
                        calibdir="/scratch/astro/nicolo.fiaba/gelsa-spectra/calib/", zero_order_catalog=None)
        # Define the frame associated to the pointing
        frame = G.load_spec_frame(**file_list[0])

        return frame
        
    def fit_detector(self, det_n, image=None, frame=None):
        # Fit the relation for one single detector: 0 <= det_n < 16
        if not (0 <= det_n < 16):
            print("Detector number not valid\n")
            return
            
        if image is None or frame is None or self.pt_id is not None:
            frame = self.load_data_to_frame()   
            print("\nCurrently fitting for detector {} in pointing n° {}".format(det_n, self.pt_id))
            # - Read the sources in chosen detector
            im, var, mask = frame.get_detector(det_n)
            im_ma = np.ma.array(im, mask=mask) # images of data in detector "det_n"
        else:
            print("\nFitting using provided image data (no loading).")
            
        # - Read the same sources from the 2MASS catalog
        x1,y1,det1 = frame.radec_to_pixel(self.cat['RAJ2000'], self.cat['DEJ2000'], wavelength=12000)
        x2,y2,det2 = frame.radec_to_pixel(self.cat['RAJ2000'], self.cat['DEJ2000'], wavelength=19000)

        # Take sources for which the left end OR right end falls in the detector det_n
        on_detector = (det1 == det_n) | (det2 == det_n)
        x1 = x1[on_detector]
        y1 = y1[on_detector]
        x2 = x2[on_detector]
        y2 = y2[on_detector]
        # ------------------------------------------------------------------------------------------------
        
        ra_on_detector = self.cat['RAJ2000'][on_detector]
        dec_on_detector = self.cat['DEJ2000'][on_detector]
        
        # X-variable for the fit: MAGNITUDE in the J band 
        jmag_on_detector = self.cat['Jmag'][on_detector]
            
        # Now x1, y1, x2, y2 contain the left-right edges of the objects we have in the frame
        
        # ------------------------------------------------------------------------------------------------            
        # Compute the 1-d spectrum for each source
        on_det_data = self.cat[on_detector]
        spectra = np.zeros((len(on_det_data), 20))
        for el in tqdm(range(len(on_det_data))):
            im, var, norm, pix_bins = frame.resample_on_wavelength(ra_on_detector[el], 
                                                                   dec_on_detector[el], 
                                                                   wave_range=frame.params['wavelength_range'], 
                                                                   super_sample=1,
                                                                   extraction_window_pix = 20)
            # Compute the 1D spectrum
            spectra[el, :] = im.sum(axis=1)
            
        # Now we compute the FWHM of the main peak of every source 
        # This is the Y-variable for the fit
        fwhm = []
        jmag_valid = []
        
        for i, av in enumerate(spectra):
            peaks, _ = find_peaks(av)

            """
            If no peak is found, we skip that source. 
            jmag_valid is defined to make sure the size of the two arrays remains the same.
            """
            
            if len(peaks) == 0:
                continue  
            results_half = peak_widths(av, peaks, rel_height=0.5)
            max_peak_fwhm = results_half[0][np.argmax(av[peaks])]
            fwhm.append(max_peak_fwhm)
            jmag_valid.append(jmag_on_detector[i])
    
        fwhm = np.array(fwhm)
        jmag_valid = np.array(jmag_valid)
        # Now we do the Linear Fit
        reg = LinearRegression().fit(np.array(jmag_valid).reshape(-1, 1), fwhm)

        # Return the two fit coefficients. This is for ONE DETECTOR!
        return reg.coef_[0], reg.intercept_

    """
    Loop the previous function over the 16 detectors in order to compute (16, 2) coefficients for the full frame
    """
    def fit_full_frame(self): 
        fit_coeffs = np.zeros((16, 2))
        for j in range(16):
            m_, q_ = self.fit_detector(det_n = j)
            fit_coeffs[j, :] = m_, q_
            
        return fit_coeffs
