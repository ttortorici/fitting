import numpy as np
from itertools import product
from scipy.optimize import least_squares
import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams["mathtext.fontset"] = "cm"


class Background:
    def __init__(self, centers, widths):
        self.centers = np.array(centers).reshape((len(centers), 1))
        self.widths = np.array(widths).reshape((len(widths), 1))


class NewBackground(Background):
    def __init__(self):
        bkgd_peak_centers = [0.5747, 0.8418, 1.0727, 1.3745, 1.704, 2.0, 2.3]
        bkgd_peak_widths = [.2, .2, .2, .2, .2, .2, .2]
        super().__init__(bkgd_peak_centers, bkgd_peak_widths)


class Hexagonal:

    def __init__(self, a, c, q, counts, name, weights=None, wavelength=1.54185, background="new", det_dist=150):
        self.free_param_num = 0
        self.wavelength = wavelength
        self.q = q
        self.counts = counts
        self.name = name

        self.a = a
        self.c = c
        self.det_dist = det_dist
        self.strain_lock = False

        if weights is None:
            self.weights = np.sqrt(counts)

        if background == "new":
            self.bkgd = NewBackground()
            self.bkgd_heights = np.zeros(self.bkgd.centers.shape)
            self.bkgd_const = 1
            self.bkgd_lin = 0
        
        self.show_data()
        print("In next cell:")
        print(" - Set background constant with .set_background_constant(background_constant)")
        print(" - Set q_min and q_max to the desired range of q values to fit using `.set_range(q_min, q_max)`.")
    
    def set_background_constant(self, background_constant, background_linear=0):
        self.bkgd_const = background_constant
        self.bkgd_lin = background_linear
        print("Keep adjusting and re-running this cell until the fit looks good.\n")
        print("In next cell:")
        print(f" - Set background with `.set_background([len {len(self.bkgd_heights)} array])`.\n")
    
    def set_background(self, background_heights):
        self.bkgd_heights = np.array(background_heights).reshape((len(background_heights), 1))
        self.show_fit()
        print("Keep adjusting and re-running this cell until the fit looks good.")
        print(f"The peak centers are at: {self.bkgd.centers.flatten()}\n")
        print("In next cell:")
        print(" - Fit peak heights with .fit_peak_heights.")
    
    def fit_peak_heights(self):
        print("In next cell:")
        print(" - Fit lattice parameters and peak heights with `.fit_lattice_parameters()`.\n")
        free_parameters = list(self.peak_heights.flatten())
        self.free_param_num = len(free_parameters)
        def residuals(params):
            peak_heights = np.array(params).reshape((len(params), 1))
            counts_hat = self.fitting_function(self.a, self.c, self.w0, peak_heights,
                                               self.grain_size, self.voight, self.strain,
                                               self.goniometer_offset, self.det_dist,
                                               self.bkgd_heights, self.bkgd_const, self.bkgd_lin)
            return (counts_hat - self.counts) / self.weights

        fit = least_squares(residuals, free_parameters, method="lm")
        
        self.report_fit(fit)
        std = np.sqrt(np.diagonal(np.linalg.inv(fit.jac.T @ fit.jac) * (fit.fun.T @ fit.fun / (fit.fun.size - fit.x.size))))
        for ii in range(len(self.hkl)):
            print(f'({self.hkl[ii, 0]},{self.hkl[ii, 1]},{self.hkl[ii, 2]}): {fit.x[ii]:.5f} \u00B1 {std[ii]:.5f}')
        self.peak_heights = fit.x.reshape((len(fit.x), 1))
        self.show_fit()

    def fit_lattice_parameters(self):
        print("In next cell:")
        print(" - Fit peak widths with .fit_peak_widths().\n")
        free_parameters = [self.a, self.c] + list(self.peak_heights.flatten())
        self.free_param_num = len(free_parameters)
        def residuals(params):
            a, c = params[:2]
            peak_heights = np.array(params[2:]).reshape((len(self.peak_heights), 1))
            counts_hat = self.fitting_function(a, c, self.w0, peak_heights,
                                               self.grain_size, self.voight, self.strain,
                                               self.goniometer_offset, self.det_dist,
                                               self.bkgd_heights, self.bkgd_const, self.bkgd_lin)
            return (counts_hat - self.counts) / self.weights
        
        fit = least_squares(residuals, free_parameters, method="lm")

        std = np.sqrt(np.diagonal(np.linalg.inv(fit.jac.T @ fit.jac) * (fit.fun.T @ fit.fun / (fit.fun.size - fit.x.size))))
        print(f"a = ({fit.x[0]:.4f} \u00B1 {std[0]:.4f}) inv Angstroms")
        print(f"c = ({fit.x[1]:.4f} \u00B1 {std[1]:.4f}) inv Angstroms")
        
        self.report_fit(fit)
        self.a, self.c = fit.x[:2]
        self.peak_heights = fit.x[2:].reshape((len(self.peak_heights), 1))
        self.report_peaks()
        self.show_fit()

    def fit_peak_widths(self):
        print("In next cell:")
        print(" - Fit peak shapes with `.fit_strain()`.\n")
        free_parameters = [self.a, self.c, self.grain_size] + list(self.peak_heights.flatten())
        self.free_param_num = len(free_parameters)
        def residuals(params):
            a, c, grain_size = params[:3]
            peak_heights = np.array(params[3:]).reshape((len(self.peak_heights), 1))
            counts_hat = self.fitting_function(a, c, self.w0, peak_heights,
                                               grain_size, self.voight, self.strain,
                                               self.goniometer_offset, self.det_dist,
                                               self.bkgd_heights, self.bkgd_const, self.bkgd_lin)
            return (counts_hat - self.counts) / self.weights
        
        fit = least_squares(residuals, free_parameters, method="lm")

        std = np.sqrt(np.diagonal(np.linalg.inv(fit.jac.T @ fit.jac) * (fit.fun.T @ fit.fun / (fit.fun.size - fit.x.size))))
        print(f"a = ({fit.x[0]:.4f} \u00B1 {std[0]:.4f}) inv Angstroms")
        print(f"c = ({fit.x[1]:.4f} \u00B1 {std[1]:.4f}) inv Angstroms")
        print(f"grain size = ({fit.x[2]:.4f} \u00B1 {std[2]:.4f})")

        self.report_fit(fit)
        self.a, self.c, self.grain_size = fit.x[:3]
        self.peak_heights = fit.x[3:].reshape((len(self.peak_heights), 1))
        self.report_peaks()
        self.show_fit()
    
    def fit_strain(self):
        self.strain_lock = False
        print("In next cell:")
        print(" - Fit goniometer offset with `.fit_voight()`.\n")
        # free_parameters = [self.a, self.c, self.grain_size, self.strain] + list(self.peak_heights.flatten())
        free_parameters = [self.a, self.c, self.strain] + list(self.peak_heights.flatten())
        self.free_param_num = len(free_parameters)
        def residuals(params):
            # a, c, grain_size, strain = params[:4]
            a, c, strain = params[:3]
            peak_heights = np.array(params[3:]).reshape((len(self.peak_heights), 1))
            counts_hat = self.fitting_function(a, c, self.w0, peak_heights,
                                               self.grain_size, self.voight, strain,
                                               # grain_size, self.voight, strain,
                                               self.goniometer_offset, self.det_dist,
                                               self.bkgd_heights, self.bkgd_const, self.bkgd_lin)
            return (counts_hat - self.counts) / self.weights
        
        fit = least_squares(residuals, free_parameters, method="lm")

        std = np.sqrt(np.diagonal(np.linalg.inv(fit.jac.T @ fit.jac) * (fit.fun.T @ fit.fun / (fit.fun.size - fit.x.size))))
        print(f"a = ({fit.x[0]:.4f} \u00B1 {std[0]:.4f}) inv Angstroms")
        print(f"c = ({fit.x[1]:.4f} \u00B1 {std[1]:.4f}) inv Angstroms")
        # print(f"grain size = ({fit.x[2]:.4f} \u00B1 {std[2]:.4f})")
        print(f"strain = ({fit.x[2]:.6f} \u00B1 {std[3]:.6f})")

        if fit.x[2] < std[2]:
            print("Strain is negative or less than it's uncertainty. Rejecting this fit and locking strain to 0.")
            self.strain_lock = True
        else:
            self.report_fit(fit)
            self.a, self.c, self.strain = fit.x[:3]
            self.peak_heights = fit.x[3:].reshape((len(self.peak_heights), 1))
            self.report_peaks()
            self.show_fit()
    
    def fit_voight(self):
        print("In next cell:")
        print(" - Fit everything with the background free `.fit_full()`.\n")
        free_parameters = [self.a, self.c, self.grain_size, self.voight]

        if not self.strain_lock:
            free_parameters.append(self.strain)
        free_parameters.extend(list(self.peak_heights.flatten()))
        self.free_param_num = len(free_parameters)
        if self.strain_lock:
            def residuals(params):
                a, c, grain_size, voight = params[:4]
                peak_heights = np.array(params[4:]).reshape((len(self.peak_heights), 1))
                counts_hat = self.fitting_function(a, c, self.w0, peak_heights,
                                                   grain_size, voight, 0,
                                                   self.goniometer_offset, self.det_dist,
                                                   self.bkgd_heights, self.bkgd_const, self.bkgd_lin)
                return (counts_hat - self.counts) / self.weights
        else:
            def residuals(params):
                a, c, grain_size, voight, strain = params[:5]
                peak_heights = np.array(params[5:]).reshape((len(self.peak_heights), 1))
                counts_hat = self.fitting_function(a, c, self.w0, peak_heights,
                                                   grain_size, voight, strain,
                                                   self.goniometer_offset, self.det_dist,
                                                   self.bkgd_heights, self.bkgd_const, self.bkgd_lin)
                return (counts_hat - self.counts) / self.weights
        
        fit = least_squares(residuals, free_parameters, method="lm", ftol=1e-9, xtol=1e-9, gtol=1e-9)

        std = np.sqrt(np.diagonal(np.linalg.inv(fit.jac.T @ fit.jac) * (fit.fun.T @ fit.fun / (fit.fun.size - fit.x.size))))
        print(f"a = ({fit.x[0]:.4f} \u00B1 {std[0]:.4f}) inv Angstroms")
        print(f"c = ({fit.x[1]:.4f} \u00B1 {std[1]:.4f}) inv Angstroms")
        print(f"grain size = ({fit.x[2]:.4f} \u00B1 {std[2]:.4f})")
        print(f"voight = ({fit.x[3]:.4f} \u00B1 {std[3]:.4f})")
        if self.strain_lock:
            print(f"strain locked at 0")
        else:
            print(f"strain = ({fit.x[4]:.4f} \u00B1 {std[4]:.4f})")

        if not (fit.x[3] >= 0 and fit.x[3] <= 1):
            print("Voight parameter outside of bounds. Rejecting fit and setting voight to 0.5.")
            self.voight = 0.5
        else:
            self.report_fit(fit)
            self.a, self.c, self.grain_size, self.voight = fit.x[:4]
            if self.strain_lock:
                self.peak_heights = fit.x[4:].reshape((len(self.peak_heights), 1))
            else:
                self.strain = fit.x[4]
                self.peak_heights = fit.x[5:].reshape((len(self.peak_heights), 1))
            self.report_peaks()
            self.show_fit()
    
    def fit_full(self):
        free_parameters = [self.a, self.c, self.grain_size, self.voight]
        if not self.strain_lock:
            free_parameters.append(self.strain)
        free_parameters.extend(list(self.peak_heights.flatten()))
        free_parameters.extend(list(self.bkgd_heights.flatten()))
        free_parameters.append(self.bkgd_const)
        self.free_param_num = len(free_parameters)

        if self.strain_lock:
            def residuals(params):
                a, c, grain_size, voight = params[:4]
                peak_heights = np.array(params[4:4+len(self.peak_heights)]).reshape((len(self.peak_heights), 1))
                bkgd_heights = np.array(params[4+len(self.peak_heights):4+len(self.peak_heights)+len(self.bkgd_heights)]).reshape((len(self.bkgd_heights), 1))
                bkgd_const = params[4+len(self.peak_heights)+len(self.bkgd_heights)]
                counts_hat = self.fitting_function(a, c, self.w0, peak_heights,
                                                   grain_size, voight, 0,
                                                   self.goniometer_offset, self.det_dist,
                                                   bkgd_heights, bkgd_const, self.bkgd_lin)
                return (counts_hat - self.counts) / self.weights
        else:
            def residuals(params):
                a, c, grain_size, voight, strain = params[:5]
                peak_heights = np.array(params[5:5+len(self.peak_heights)]).reshape((len(self.peak_heights), 1))
                bkgd_heights = np.array(params[5+len(self.peak_heights):5+len(self.peak_heights)+len(self.bkgd_heights)]).reshape((len(self.bkgd_heights), 1))
                bkgd_const = params[5+len(self.peak_heights)+len(self.bkgd_heights)]
                counts_hat = self.fitting_function(a, c, self.w0, peak_heights,
                                                   grain_size, voight, strain,
                                                   self.goniometer_offset, self.det_dist,
                                                   bkgd_heights, bkgd_const, self.bkgd_lin)
                return (counts_hat - self.counts) / self.weights

        fit = least_squares(residuals, free_parameters, method="lm", ftol=1e-9, xtol=1e-9, gtol=1e-9)

        std = np.sqrt(np.diagonal(np.linalg.inv(fit.jac.T @ fit.jac) * (fit.fun.T @ fit.fun / (fit.fun.size - fit.x.size))))
        print(f"a = ({fit.x[0]:.4f} \u00B1 {std[0]:.4f}) inv Angstroms")
        print(f"c = ({fit.x[1]:.4f} \u00B1 {std[1]:.4f}) inv Angstroms")
        print(f"grain size = ({fit.x[2]:.4f} \u00B1 {std[2]:.4f})")
        print(f"voight = ({fit.x[3]:.4f} \u00B1 {std[3]:.4f})")
        if self.strain_lock:
            print(f"strain locked at 0")
            for ii in range(len(self.hkl)):
                print(f'({self.hkl[ii, 0]},{self.hkl[ii, 1]},{self.hkl[ii, 2]}): {fit.x[ii+4]:.5f} \u00B1 {std[ii+4]:.5f}')
            for ii in range(len(self.bkgd_heights)):
                print(f"bkgd peak {ii}: {fit.x[ii+4+len(self.hkl)]:.5f} \u00B1 {std[ii+4+len(self.hkl)]:.5f}")
            print(f"background constant = {fit.x[4+len(self.hkl)+len(self.bkgd_heights)]:.5f} \u00B1 {std[4+len(self.hkl)+len(self.bkgd_heights)]:.5f}")
        else:
            print(f"strain = ({fit.x[4]:.4f} \u00B1 {std[4]:.4f})")
            for ii in range(len(self.hkl)):
                print(f'({self.hkl[ii, 0]},{self.hkl[ii, 1]},{self.hkl[ii, 2]}): {fit.x[ii+5]:.5f} \u00B1 {std[ii+5]:.5f}')
            for ii in range(len(self.bkgd_heights)):
                print(f"bkgd peak {ii}: {fit.x[ii+5+len(self.hkl)]:.5f} \u00B1 {std[ii+5+len(self.hkl)]:.5f}")
            print(f"background constant = {fit.x[5+len(self.hkl)+len(self.bkgd_heights)]:.5f} \u00B1 {std[5+len(self.hkl)+len(self.bkgd_heights)]:.5f}")
            
        self.report_fit(fit)
        self.a, self.c, self.grain_size, self.voight, self.goniometer_offset = fit.x[:5]
        if self.strain_lock:
            self.peak_heights = fit.x[4:4+len(self.peak_heights)].reshape((len(self.peak_heights), 1))
            self.bkgd_heights = fit.x[4+len(self.peak_heights):4+len(self.peak_heights)+len(self.bkgd_heights)].reshape((len(self.bkgd_heights), 1))
            self.bkgd_const = fit.x[4+len(self.peak_heights)+len(self.bkgd_heights)]
        else:
            self.strain = fit.x[4]
            self.peak_heights = fit.x[5:5+len(self.peak_heights)].reshape((len(self.peak_heights), 1))
            self.bkgd_heights = fit.x[5+len(self.peak_heights):5+len(self.peak_heights)+len(self.bkgd_heights)].reshape((len(self.bkgd_heights), 1))
            self.bkgd_const = fit.x[5+len(self.peak_heights)+len(self.bkgd_heights)]
        self.report_peaks()
        self.show_fit()

    
    def report_peaks(self):
        q = 2 * np.pi * np.sqrt(self.hh_hk_kk / (self.a * self.a) + self.l_sq / (self.c * self.c))
        warnings = 0
        for ii in range(len(self.hkl)):
            if self.peak_heights[ii] < 0.001:
                warnings += 1
                print(f"Warning: ({self.hkl[ii, 0]},{self.hkl[ii, 1]},{self.hkl[ii, 2]}) at q = {q[ii, 0]:.5f}: {self.peak_heights[ii, 0]:.5f}")
        if warnings:
            print("Remove peaks considering forbidden conditions (h+k=3n & l is odd) near the same q of these.")
            print("Use .remove_peak(h, k, l) to remove a peak. Do this above .fit_peak_heights() and re-run Notebook.")
            print("")
        for ii in range(len(self.hkl)):
            print(f'({self.hkl[ii, 0]},{self.hkl[ii, 1]},{self.hkl[ii, 2]}) at q = {q[ii, 0]:.5f} inv A: {self.peak_heights[ii, 0]:.5f}')

    def report_fit(self, fit):
        dof = len(self.counts) - len(fit.x)
        reduced_chi_sq = 2 * fit.cost / dof
        print(f"Reduced chi-squared: {reduced_chi_sq:.3f}")
        print(f"message: {fit.message}")
        print(f"Number of function evaluations: {fit.nfev}")
        print(f"Number of Jacobian evaluations: {fit.njev}\n")
    
    def chi_sq(self):
        counts_hat = self.fitting_function(self.a, self.c, self.w0, self.peak_heights, self.grain_size, self.voight, self.strain,
                                           self.goniometer_offset, self.det_dist, self.bkgd_heights, self.bkgd_const, self.bkgd_lin)
        return np.sum(((self.counts - counts_hat) / self.weights) ** 2)

    def report_chi_sq(self):
        chi_sq = self.chi_sq()
        print(f"Chi-squared: {chi_sq:.3f}")
        dof = len(self.counts) - self.free_param_num
        print(f"Degrees of freedom: {dof}")
        print(f"Reduced chi-squared: {chi_sq / dof:.3f}")
        

    def initialize_fit(self, q_buffer=1.0):
        self.lorentz = self.lorentz_polarization_factor()

        self.w0 = 0.003
        self.grain_size = 660
        self.voight = 0.5
        self.strain = 0
        self.goniometer_offset = 0

        self.multiplicity = []
        self.hkl = []
        self.q_center = []
        for h, k, l in product(range(5), repeat=3):
            if h + k + l > 0 and h <= k:
                q = 2 * np.pi * np.sqrt(4.0 / 3.0 * (h * h + h * k + k * k) / (self.a * self.a) + l * l / (self.c * self.c))
                if self.q.min() <= q <= self.q.max() * q_buffer:
                    self.q_center.append(q)
                    self.hkl.append([h, k, l])

                    if h + k == 0:
                        self.multiplicity.append(2)
                    elif l == 0 and (h == 0 or h == k):
                        self.multiplicity.append(6)
                    elif h == k or l == 0 or h == 0:
                        self.multiplicity.append(12)
                    else:
                        self.multiplicity.append(24)
        self.hkl = np.array(self.hkl)
        self.q_center = np.array(self.q_center)
        self.multiplicity = np.array(self.multiplicity)
        self.peak_heights = np.ones((len(self.hkl), 1))

        sort_ind = np.argsort(self.q_center)
        self.q_center = self.q_center[sort_ind].reshape((len(self.q_center), 1))
        self.hkl = self.hkl[sort_ind]
        self.multiplicity = self.multiplicity[sort_ind].reshape((len(self.multiplicity), 1))

        self.hh_hk_kk = 4. / 3. * (self.hkl[:, 0] * self.hkl[:, 0] + self.hkl[:, 0] * self.hkl[:, 1] + self.hkl[:, 1] * self.hkl[:, 1]).reshape((self.hkl.shape[0], 1))
        self.l_sq = (self.hkl[:, 2] * self.hkl[:, 2]).reshape((self.hkl.shape[0], 1))

        print('Number of possible reflections within data: %d' % len(self.hkl))
        for ii in range(len(self.hkl)):
            print(f'hkl: ({self.hkl[ii, 0]},{self.hkl[ii, 1]},{self.hkl[ii, 2]}), q: {self.q_center[ii, 0]:.3f}, multiplicity: {self.multiplicity[ii, 0]}')

    def remove_peak(self, h, k, l):
        ind = np.where((self.hkl[:, 0] == h) & (self.hkl[:, 1] == k) & (self.hkl[:, 2] == l))[0]
        if len(ind) == 1:
            self.hkl = np.delete(self.hkl, ind, axis=0)
            self.multiplicity = np.delete(self.multiplicity, ind, axis=0)
            self.peak_heights = np.delete(self.peak_heights, ind, axis=0)
            self.hh_hk_kk = np.delete(self.hh_hk_kk, ind, axis=0)
            self.l_sq = np.delete(self.l_sq, ind, axis=0)
        else:
            print("Peak already is not present.")

    def add_peak(self, h, k, l):
        self.hkl = np.vstack((self.hkl, [h, k, l]))
        if h + k == 0:
            self.multiplicity = np.vstack((self.multiplicity, [2]))
        elif l == 0 and (h == 0 or h == k):
            self.multiplicity.append(6)
        elif h == k or l == 0 or h == 0:
            self.multiplicity.append(12)
        else:
            self.multiplicity.append(24)

    def lorentz_polarization_factor(self):
        """calculate the lorentz polarization factor"""
        sintheta = self.wavelength * self.q / (4 * np.pi)
        theta = np.arcsin(sintheta)
        cos2theta = np.cos(2 * theta)
        return (1 + cos2theta * cos2theta) / (sintheta * sintheta * np.cos(theta))

    def calc_background(self, bkgd_heights):
        """Calculate the background pre-lorentz factor"""
        arg = (self.q - self.bkgd.centers) / self.bkgd.widths
        return np.sum(bkgd_heights * np.exp(-0.5 * arg * arg), axis=0)
    
    def show_data(self):
        plt.figure(figsize=(3.5, 3))
        ax = plt.subplot()
        ax.scatter(
            self.q, self.counts,
            s=5,  # marker size
            marker="o",  # marker shape
            edgecolors="black",  # marker edge color
            lw=.75,  # marker edge width
            alpha=1,  # transparency
            facecolor='w'  # marker face color
        )
        ax.set_title(self.name)
        ax.set_xlabel(r"$q\ (\mathregular{\AA}^{-1})$")
        ax.set_ylabel("Counts")
        ax.set_yscale("log")
        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax.grid()
        plt.tight_layout()
    
    def show_fit(self):
        fit = self.fitting_function(self.a, self.c, self.w0, self.peak_heights, self.grain_size, self.voight, self.strain,
                                    self.goniometer_offset, self.det_dist, self.bkgd_heights, self.bkgd_const, self.bkgd_lin)
        self.show_data()
        plt.plot(self.q, fit, label="Fit", color="red", lw="0.5")
    
    def set_range(self, q_min, q_max, q_buffer=1.0):
        self.q_min = q_min
        self.q_max = q_max
        self.counts = self.counts[(self.q > q_min) & (self.q < q_max)]
        self.weights = self.weights[(self.q > q_min) & (self.q < q_max)]
        self.q = self.q[(self.q > q_min) & (self.q < q_max)]
        self.initialize_fit(q_buffer)
        self.show_fit()
    
    def fitting_function(self, a, c, w0, peak_heights, grain_size, voight, strain, goniometer_offset, det_dist, bkgd_heights, bkgd_const, bkgd_lin):
        inv_d = np.sqrt(self.hh_hk_kk / (a * a) + self.l_sq / (c * c))
        theta1 = np.arcsin(0.5 * self.wavelength * inv_d)
        wq = w0 + (0.9 * self.wavelength / (grain_size * np.cos(theta1))) + strain * np.tan(theta1)
        q_c = 4 * np.pi / self.wavelength * np.sin(theta1)
        arg = (self.q - q_c) / wq
        arg_sq = arg * arg

        peaks = np.sum(
            self.multiplicity * peak_heights * ((1 - voight) * np.exp(-0.5 * arg_sq) + (voight / (arg_sq + 1))),
            axis=0
        )
        return self.lorentz * (peaks + self.calc_background(bkgd_heights)) + bkgd_const + bkgd_lin * self.q
        

class HexagonalGI(Hexagonal):
    def fitting_function(self, a, c, w0, peak_heights, grain_size, voight, strain, goniometer_offset, det_dist, bkgd_heights, bkgd_const, bkgd_lin):
        inv_d = np.sqrt(self.hh_hk_kk / (a * a) + self.l_sq / (c * c))
        theta1 = np.arcsin(0.5 * self.wavelength * inv_d)
        wq = w0 + (0.9 * self.wavelength / (grain_size * np.cos(theta1))) + strain * np.tan(theta1)
        q_c = 4 * np.pi / self.wavelength * np.sin(theta1 + goniometer_offset * np.cos(theta1) / det_dist)
        arg = (self.q - q_c) / wq
        arg_sq = arg * arg

        peaks = np.sum(
            self.multiplicity * peak_heights * ((1 - voight) * np.exp(-0.5 * arg_sq) + (voight / (arg_sq + 1))),
            axis=0
        )
        return self.lorentz * (peaks + self.calc_background(bkgd_heights)) + bkgd_const + bkgd_lin * self.q

    def fit_goniometer_offset(self):
        print("In next cell:")
        print(" - Try fitting the voight parameter with `.fit_voight()`.\n")
        free_parameters = [self.a, self.c, self.grain_size, self.goniometer_offset]
        if not self.strain_lock:
            free_parameters.append(self.strain)
        free_parameters.extend(list(self.peak_heights.flatten()))
        def residuals(params):
            a, c, grain_size, goniometer_offset = params[:4]
            if self.strain_lock:
                strain = 0
                peak_heights = np.array(params[4:]).reshape((len(self.peak_heights), 1))
            else:
                strain = params[4]
                peak_heights = np.array(params[5:]).reshape((len(self.peak_heights), 1))
            counts_hat = self.fitting_function(a, c, self.w0, peak_heights,
                                               grain_size, self.voight, strain,
                                               goniometer_offset, self.det_dist,
                                               self.bkgd_heights, self.bkgd_const, self.bkgd_lin)
            return (counts_hat - self.counts) / self.weights
        
        fit = least_squares(residuals, free_parameters, method="lm")

        std = np.sqrt(np.diagonal(np.linalg.inv(fit.jac.T @ fit.jac) * (fit.fun.T @ fit.fun / (fit.fun.size - fit.x.size))))
        print(f"a = ({fit.x[0]:.4f} \u00B1 {std[0]:.4f}) inv Angstroms")
        print(f"c = ({fit.x[1]:.4f} \u00B1 {std[1]:.4f}) inv Angstroms")
        print(f"grain size = ({fit.x[2]:.4f} \u00B1 {std[2]:.4f})")
        print(f"goniometer offset = ({fit.x[3]:.4f} \u00B1 {std[3]:.4f})")
        if self.strain_lock:
            print(f"strain locked at 0")
        else:
            print(f"strain = ({fit.x[3]:.4f} \u00B1 {std[3]:.4f})")

        self.report_fit(fit)
        self.a, self.c, self.grain_size, self.goniometer_offset = fit.x[:4]
        if self.strain_lock:
            self.peak_heights = fit.x[4:].reshape((len(self.peak_heights), 1))
        else:
            self.strain = fit.x[4]
            self.peak_heights = fit.x[5:].reshape((len(self.peak_heights), 1))
        self.report_peaks()
        self.show_fit()
    
    def fit_voight(self):
        print("In next cell:")
        print(" - Fit everything with the background free `.fit_full()`.\n")
        free_parameters = [self.a, self.c, self.grain_size, self.voight, self.goniometer_offset]
        if not self.strain_lock:
            free_parameters.append(self.strain)
        free_parameters.extend(list(self.peak_heights.flatten()))
        def residuals(params):
            a, c, grain_size, voight, goniometer_offset = params[:5]
            if self.strain_lock:
                strain = 0
                peak_heights = np.array(params[5:]).reshape((len(self.peak_heights), 1))
            else:
                strain = params[5]
                peak_heights = np.array(params[6:]).reshape((len(self.peak_heights), 1))
            counts_hat = self.fitting_function(a, c, self.w0, peak_heights,
                                               grain_size, voight, strain,
                                               goniometer_offset, self.det_dist,
                                               self.bkgd_heights, self.bkgd_const, self.bkgd_lin)
            return (counts_hat - self.counts) / self.weights
        
        fit = least_squares(residuals, free_parameters, method="lm")

        std = np.sqrt(np.diagonal(np.linalg.inv(fit.jac.T @ fit.jac) * (fit.fun.T @ fit.fun / (fit.fun.size - fit.x.size))))
        print(f"a = ({fit.x[0]:.4f} \u00B1 {std[0]:.4f}) inv Angstroms")
        print(f"c = ({fit.x[1]:.4f} \u00B1 {std[1]:.4f}) inv Angstroms")
        print(f"grain size = ({fit.x[2]:.4f} \u00B1 {std[2]:.4f})")
        print(f"voight = ({fit.x[3]:.4f} \u00B1 {std[3]:.4f})")
        print(f"goniometer offset = ({fit.x[4]:.4f} \u00B1 {std[4]:.4f})")
        if self.strain_lock:
            print(f"strain locked at 0")
        else:
            print(f"strain = ({fit.x[5]:.4f} \u00B1 {std[5]:.4f})")

        if not (fit.x[3] >= 0 and fit.x[3] <= 1):
            print("Voight parameter outside of bounds. Rejecting fit and setting voight to 0.5.")
            self.voight = 0.5
        else:
            self.report_fit(fit)
            self.a, self.c, self.grain_size, self.voight, self.goniometer_offset = fit.x[:5]
            if self.strain_lock:
                self.peak_heights = fit.x[5:].reshape((len(self.peak_heights), 1))
            else:
                self.strain = fit.x[5]
                self.peak_heights = fit.x[6:].reshape((len(self.peak_heights), 1))
            self.report_peaks()
            self.show_fit()
    
    def fit_full(self, linear_background=False):

        free_parameters = [self.a, self.c, self.grain_size, self.voight, self.goniometer_offset]
        if not self.strain_lock:
            free_parameters.append(self.strain)
        free_parameters.extend(list(self.peak_heights.flatten()))
        free_parameters.extend(list(self.bkgd_heights.flatten()))
        free_parameters.append(self.bkgd_const)
        if linear_background:
            free_parameters.append(self.bkgd_lin)
        def residuals(params):
            a, c, grain_size, voight, goniometer_offset = params[:5]
            if self.strain_lock:
                strain = 0
                peak_heights = np.array(params[5:5+len(self.peak_heights)]).reshape((len(self.peak_heights), 1))
                bkgd_heights = np.array(params[5+len(self.peak_heights):5+len(self.peak_heights)+len(self.bkgd_heights)]).reshape((len(self.bkgd_heights), 1))
                bkgd_const = params[5+len(self.peak_heights)+len(self.bkgd_heights)]
                if linear_background:
                    bkgd_lin = params[6+len(self.peak_heights)+len(self.bkgd_heights)]
                else:
                    bkgd_lin = 0
            else:
                strain = params[5]
                peak_heights = np.array(params[6:6+len(self.peak_heights)]).reshape((len(self.peak_heights), 1))
                bkgd_heights = np.array(params[6+len(self.peak_heights):6+len(self.peak_heights)+len(self.bkgd_heights)]).reshape((len(self.bkgd_heights), 1))
                bkgd_const = params[6+len(self.peak_heights)+len(self.bkgd_heights)]
                if linear_background:
                    bkgd_lin = params[7+len(self.peak_heights)+len(self.bkgd_heights)]
                else:
                    bkgd_lin = 0
            counts_hat = self.fitting_function(a, c, self.w0, peak_heights,
                                               grain_size, voight, strain,
                                               goniometer_offset, self.det_dist,
                                               bkgd_heights, bkgd_const, bkgd_lin)
            return (counts_hat - self.counts) / self.weights

        fit = least_squares(residuals, free_parameters, method="lm", ftol=1e-9, xtol=1e-9, gtol=1e-9)

        std = np.sqrt(np.diagonal(np.linalg.inv(fit.jac.T @ fit.jac) * (fit.fun.T @ fit.fun / (fit.fun.size - fit.x.size))))
        print(f"a = ({fit.x[0]:.4f} \u00B1 {std[0]:.4f}) inv Angstroms")
        print(f"c = ({fit.x[1]:.4f} \u00B1 {std[1]:.4f}) inv Angstroms")
        print(f"grain size = ({fit.x[2]:.4f} \u00B1 {std[2]:.4f})")
        print(f"voight = ({fit.x[3]:.4f} \u00B1 {std[3]:.4f})")
        print(f"goniometer offset = ({fit.x[4]:.4f} \u00B1 {std[4]:.4f})")
        if self.strain_lock:
            print(f"strain locked at 0")
            for ii in range(len(self.hkl)):
                print(f'({self.hkl[ii, 0]},{self.hkl[ii, 1]},{self.hkl[ii, 2]}): {fit.x[ii+5]:.5f} \u00B1 {std[ii+5]:.5f}')
            for ii in range(len(self.bkgd_heights)):
                print(f"bkgd peak {ii}: {fit.x[ii+5+len(self.hkl)]:.5f} \u00B1 {std[ii+5+len(self.hkl)]:.5f}")
            print(f"background constant = {fit.x[5+len(self.hkl)+len(self.bkgd_heights)]:.5f} \u00B1 {std[5+len(self.hkl)+len(self.bkgd_heights)]:.5f}")
            if linear_background:
                print(f"background linear = {fit.x[6+len(self.hkl)+len(self.bkgd_heights)]:.5f} \u00B1 {std[6+len(self.hkl)+len(self.bkgd_heights)]:.5f}")
        else:
            print(f"strain = ({fit.x[5]:.4f} \u00B1 {std[5]:.4f})")
            for ii in range(len(self.hkl)):
                print(f'({self.hkl[ii, 0]},{self.hkl[ii, 1]},{self.hkl[ii, 2]}): {fit.x[ii+6]:.5f} \u00B1 {std[ii+6]:.5f}')
            for ii in range(len(self.bkgd_heights)):
                print(f"bkgd peak {ii}: {fit.x[ii+6+len(self.hkl)]:.5f} \u00B1 {std[ii+6+len(self.hkl)]:.5f}")
            print(f"background constant = {fit.x[6+len(self.hkl)+len(self.bkgd_heights)]:.5f} \u00B1 {std[6+len(self.hkl)+len(self.bkgd_heights)]:.5f}")
            if linear_background:
                print(f"background linear = {fit.x[7+len(self.hkl)+len(self.bkgd_heights)]:.5f} \u00B1 {std[7+len(self.hkl)+len(self.bkgd_heights)]:.5f}")
        
        self.report_fit(fit)
        self.a, self.c, self.grain_size, self.voight, self.goniometer_offset = fit.x[:5]
        if self.strain_lock:
            self.peak_heights = fit.x[5:5+len(self.peak_heights)].reshape((len(self.peak_heights), 1))
            self.bkgd_heights = fit.x[5+len(self.peak_heights):5+len(self.peak_heights)+len(self.bkgd_heights)].reshape((len(self.bkgd_heights), 1))
            self.bkgd_const = fit.x[5+len(self.peak_heights)+len(self.bkgd_heights)]
            if linear_background:
                self.bkgd_lin = fit.x[6+len(self.peak_heights)+len(self.bkgd_heights)]
        else:
            self.strain = fit.x[5]
            self.peak_heights = fit.x[6:6+len(self.peak_heights)].reshape((len(self.peak_heights), 1))
            self.bkgd_heights = fit.x[6+len(self.peak_heights):6+len(self.peak_heights)+len(self.bkgd_heights)].reshape((len(self.bkgd_heights), 1))
            self.bkgd_const = fit.x[6+len(self.peak_heights)+len(self.bkgd_heights)]
            if linear_background:
                self.bkgd_lin = fit.x[7+len(self.peak_heights)+len(self.bkgd_heights)]
        self.report_peaks()
        self.show_fit()


def q_from_2th(two_theta, wavelength=1.54185):
    return 4 * np.pi * np.sin(two_theta * np.pi / 360.) / wavelength

