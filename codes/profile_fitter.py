# Define the Profile_Fitter class to find the best SEMIMAJOR_AXIS vs FWHM relation for a whole frame (16 detectors)
import numpy as np
import gelsa
from gelsa.sgs import datastore
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from scipy.signal import peak_widths, find_peaks

class Profile_Fitter:
    def __init__(self, pt_id_s, cat):
        self.pt_id_s = pt_id_s
        self.cat = cat

    def load_data_to_frame(self, pt_id):
        DS = datastore.DataStore(username='', password='',
                        cachedir='/scratch/astro/benjamin.granett/datastore'
                        )
        # Load the data related to the pointing_id
        file_list = DS.load_sir_pack("DR1_R1", pointing_id_list=[pt_id])
        # Create a Gelsa object
        G = gelsa.Gelsa(config_file="/scratch/astro/nicolo.fiaba/gelsa-spectra/calib/gelsa_config.json", 
                        calibdir="/scratch/astro/nicolo.fiaba/gelsa-spectra/calib/", zero_order_catalog=None)
        # Define the frame associated to the pointing
        frame = G.load_spec_frame(**file_list[0])

        return frame
        
    def smjax_range(self, smjax_data, center, num=20):
    # Finds the interval around one semi-major axis center, containing "num" objects (20 by default)
        smjax_data = np.asarray(smjax_data)
        # - Compute distance of each object from the center
        dist = np.abs(smjax_data - center)  
        # - Sort by distance and keep the "num" closest objects
        idx = np.argsort(dist)[:num]
        idx = np.sort(idx)
        # - Get the "num" closest objects to "center"
        closest_obj = smjax_data[idx]
        # - Return those and their indices
        return closest_obj, idx

    def fit_single_detector(self, det_n, smjax_centers, pt_id=None, num_per_range=20, image=None, frame=None):
        # Fit the relation for one single detector: 0 <= det_n < 16
        if not (0 <= det_n < 16):
            print("Detector number not valid\n")
            return
            
        if image is None or frame is None or pt_id is not None:
            frame = self.load_data_to_frame(pt_id)   
            print("\nCurrently fitting for detector {} in pointing n° {}".format(det_n, pt_id))
            # - Read the sources in chosen detector
            im, var, mask = frame.get_detector(det_n)
            im_ma = np.ma.array(im, mask=mask) # images of data in detector "det_n"
        else:
            print("\nFitting using provided image data (no loading).")
            
        # - Read the same sources from EDFN_deep.fits catalog
        x1,y1,det1 = frame.radec_to_pixel(self.cat['RIGHT_ASCENSION'], self.cat['DECLINATION'], wavelength=12000)
        x2,y2,det2 = frame.radec_to_pixel(self.cat['RIGHT_ASCENSION'], self.cat['DECLINATION'], wavelength=19000)
        on_detector = (det1 == det_n) & (det2 == det_n)
        # --------------------
        x1 = x1[on_detector]
        y1 = y1[on_detector]
        x2 = x2[on_detector]
        y2 = y2[on_detector]
        # --------------------
        ra_on_detector = self.cat[on_detector]['RIGHT_ASCENSION']
        dec_on_detector = self.cat[on_detector]['DECLINATION']
            # Now x1, y1, x2, y2 contain the left-right edges of the objects we have in the frame
        
        smjax_range_values = []
        indices = []
        for c in smjax_centers:
            masked, index = self.smjax_range(self.cat[on_detector]["SEMIMAJOR_AXIS"], center=c, num=num_per_range)
            smjax_range_values.append(masked)
            indices.append(index)
            
        # --------------------------------------------------------    
        # Print how many sources are in more than one interval 
        indices_set = [set(row) for row in indices]
        in_all = set.intersection(*indices_set)
        from itertools import combinations
        in_two = set()
        for s1, s2 in combinations(indices_set, 2):
            in_two.update(s1 & s2)

        # Remove those that appear in all 3
        in_two -= in_all

        print("Values in all 3 smjax ranges:", in_all)
        print("Values in exactly 2 smjax ranges:", in_two)
        # --------------------------------------------------------    
        
        # - Compute the 1-d spectrum for each source
        spectra = np.zeros((len(smjax_centers), num_per_range, 11)) # 11 pixels vertically
        for i, indix in enumerate(indices):
            temp = self.cat[on_detector][indix] # Table with the objects in the current range of semi-major axis
            for el in tqdm(range(len(temp))):
                im, var, norm, pix_bins = frame.resample_on_wavelength(temp["RIGHT_ASCENSION"][el], 
                                                                       temp["DECLINATION"][el], 
                                                                       wave_range=frame.params['wavelength_range'], 
                                                                       super_sample=1)
                spectra[i, el,:] = im.sum(axis=1)
                
        # - Compute the average spectra for the detector: one average spectrum for each semi-major axis center
        avg_spectra = spectra.mean(axis=1)
        # - Compute the FWHM of the main peak 
        fwhm = np.zeros(len(avg_spectra))
        for i, av in enumerate(avg_spectra):
            peaks, _ = find_peaks(av)
            results_half = peak_widths(av, peaks, rel_height=0.5)
            max_peak_fwhm = results_half[0][np.argmax(av[peaks])]
            fwhm[i] = max_peak_fwhm
        # - Linear fit 
        reg = LinearRegression().fit(np.array(smjax_centers).reshape(-1, 1), fwhm)

        return reg.coef_[0], reg.intercept_, fwhm, avg_spectra

    def fit_full_frame(self, smjax_centers, pt_id, num_per_range):
        fit_coeffs = np.zeros((16, 2))
        fwhm = np.zeros((16, len(smjax_centers)))
        for j in range(16):
            m_, q_, fwhm_, avg_spectra_ = self.fit_single_detector(j, smjax_centers, pt_id, num_per_range)
            fit_coeffs[j, :] = m_, q_
            fwhm[j, :] = fwhm_
            
        return fit_coeffs, fwhm

    
    
