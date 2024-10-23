from fitting.data import DataSet
from fitting.debye import Debye
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams["mathtext.fontset"] = "cm"


class Debye33(Debye):

    def __init__(self, data: DataSet, name: str = "", calibration_data: DataSet = None):
        super().__init__(data, name, calibration_data)
        self.emat0 = 3.3
        self.emat1 = 3.7e-4
        self.td0 = 1e-5
        self.td1 = 0.0
        self.td2 = 0.0
        self.num_peaks = 3
        self.num_relax_times = 11
        self.curie_temperature = 0.0
        self.ln_attempt_time1 = -30.0
        self.ln_attempt_time2 = -30.0
        self.ln_attempt_time3 = -30.0
        self.activation_energy1 = 700.0
        self.activation_energy2 = 1400.0
        self.activation_energy3 = 1700.0
        self.coupling_energy1 = 0
        self.coupling_energy2 = 0
        self.coupling_energy3 = 0
        self.populations1 = np.zeros((self.num_relax_times, 1, 1), dtype=np.float64)
        self.populations2 = np.zeros((self.num_relax_times, 1, 1), dtype=np.float64)
        self.populations3 = np.zeros((self.num_relax_times, 1, 1), dtype=np.float64)
        self.populations1[5, 0, 0] = 1.0        # 5 is the center
        self.populations2[5, 0, 0] = 1.0        # 5 is the center
        self.populations3[5, 0, 0] = 1.0        # 5 is the center
        self.peak_width1 = 50.0
        self.peak_width2 = 50.0
        self.peak_width3 = 50.0

    def initialize_fit(self):
        self.emat0 = 3.3
        self.emat1 = 3.7e-4
        self.td0 = 1e-5
        self.td1 = 0.0
        self.td2 = 0.0
        self.num_peaks = 3
        self.num_relax_times = 11
        self.curie_temperature = 0.0
        self.ln_attempt_time = -30.0
        self.activation_energy1 = 700.0
        self.activation_energy2 = 1400.0
        self.activation_energy3 = 1700.0
        self.coupling_energy1 = 0
        self.coupling_energy2 = 0
        self.coupling_energy3 = 0
        self.populations1 = np.zeros((self.num_relax_times, 1, 1), dtype=np.float64)
        self.populations2 = np.zeros((self.num_relax_times, 1, 1), dtype=np.float64)
        self.populations1[5, 0, 0] = 1.0        # 5 is the center
        self.populations2[5, 0, 0] = 1.0        # 5 is the center
        self.populations3[5, 0, 0] = 1.0
        self.peak_width1 = 50.0
        self.peak_width2 = 50.0
        self.peak_width3 = 50.0
    
    def set_population(self, peak_num: int, new_population: float, step: int=0, override: int=None):
        if peak_num == 1:
            populations = self.populations1
        elif peak_num == 2:
            populations = self.populations2
        elif peak_num == 3:
            populations = self.populations3
        else:
            raise ValueError("peak_num exceeds numbers of peaks")
        if override is None:
            if step:
                populations[5 + step, 0, 0] = new_population
                populations[5 - step, 0, 0] = new_population
            else:
                populations[5, 0, 0] = new_population
        else:
            populations[override, 0, 0] = new_population

    @classmethod
    def susceptibility1(cls, temperature: np.ndarray, coupling_energy: float):
        return cls.susceptibility_noninteracting(temperature, coupling_energy)
    
    @classmethod
    def susceptibility2(cls, temperature: np.ndarray, coupling_energy: float):
        return cls.susceptibility_noninteracting(temperature, coupling_energy)
    
    @classmethod
    def susceptibility3(cls, temperature: np.ndarray, coupling_energy: float):
        return cls.susceptibility_noninteracting(temperature, coupling_energy)
    
    def fit_1(self):
        params = [self.activation_energy1, self.activation_energy2, self.activation_energy3,
                  self.coupling_energy1, self.coupling_energy2, self.coupling_energy3,
                  self.peak_width1, self.peak_width2, self.peak_width3, self.emat0, self.emat1]
        params.extend(list(self.populations1.flatten()))
        params.extend(list(self.populations2.flatten()))
        params.extend(list(self.populations3.flatten()))

        print(params)

        def residuals(params):
            activation_energy1, activation_energy2, activation_energy3 = params[:3]
            coupling_energy1, coupling_energy2, coupling_energy3 = params[3:6]
            peak_width1, peak_width2, peak_width3 = params[6:9]
            emat0 = params[9]
            emat1 = params[10]
            populations1 = np.array(params[11:22], dtype=np.float64).reshape((11, 1, 1))
            populations2 = np.array(params[22:33], dtype=np.float64).reshape((11, 1, 1))
            populations3 = np.array(params[33:44], dtype=np.float64).reshape((11, 1, 1))
            cap, imag = self.fitting_function(self.temperature, self.angular_frequency, self.curie_temperature, self.ln_attempt_time,
                                              activation_energy1, activation_energy2, activation_energy3,
                                              coupling_energy1, coupling_energy2, coupling_energy3,
                                              populations1, populations2, populations3,
                                              peak_width1, peak_width2, peak_width3, emat0, emat1,
                                              self.bare0, self.bare1, self.bare2, self.td0, self.td1, self.td2)
            return ((self.capacitance - cap) + (self.imaginary_capacitance - imag)).flatten()
        
        fit = least_squares(residuals, params, method='lm')
        std = np.sqrt(np.diagonal(np.linalg.inv(fit.jac.T @ fit.jac) * (fit.fun.T @ fit.fun / (fit.fun.size - fit.x.size))))
        print(f"U1 = ({self.convert_energy(fit.x[0], 'kcal'):.4f} \u00B1 {self.convert_energy(std[0], 'kcal'):.4f}) kcal/mol")
        print(f"U2 = ({self.convert_energy(fit.x[1], 'kcal'):.4f} \u00B1 {self.convert_energy(std[1], 'kcal'):.4f}) kcal/mol")
        print(f"U3 = ({self.convert_energy(fit.x[2], 'kcal'):.4f} \u00B1 {self.convert_energy(std[2], 'kcal'):.4f}) kcal/mol")
        print(f"s1 = ({self.convert_energy(fit.x[3], 'kcal'):.4f} \u00B1 {self.convert_energy(std[3], 'kcal'):.4f}) kcal/mol")
        print(f"s2 = ({self.convert_energy(fit.x[4], 'kcal'):.4f} \u00B1 {self.convert_energy(std[4], 'kcal'):.4f}) kcal/mol")
        print(f"s3 = ({self.convert_energy(fit.x[5], 'kcal'):.4f} \u00B1 {self.convert_energy(std[5], 'kcal'):.4f}) kcal/mol")

        self.report_fit(fit)
        self.show_fit()

    @classmethod
    def fitting_function(cls, temperature: np.ndarray, angular_frequency, curie_temperature: float,
                         ln_attempt_time1: float, ln_attempt_time2: float, ln_attempt_time3: float,
                         activation_energy1: float, activation_energy2: float, activation_energy3: float,
                         coupling_energy1: float, coupling_energy2: float, coupling_energy3: float,
                         populations1: np.ndarray, populations2: np.ndarray, populations3: float,
                         peak_width1: float, peak_width2: float, peak_width3: float, emat0: float, emat1: float,
                         bare0: float, bare1: float, bare2: float, 
                         td0: float, td1: float, td2: float):
        offsets = np.linspace(-5, 5, 11).reshape((11, 1, 1))
        tau1 = np.exp(ln_attempt_time1 + (activation_energy1 + offsets * peak_width1) / temperature)     # shape (num_relax_times, 1, 1)
        tau2 = np.exp(ln_attempt_time2 + (activation_energy2 + offsets * peak_width2) / temperature)     # shape (num_relax_times, 1, 1)
        tau3 = np.exp(ln_attempt_time3 + (activation_energy3 + offsets * peak_width3) / temperature)

        temperature_300 = temperature - 300.0
        temperature_curie_inv = 1. / (temperature - curie_temperature)
        omega_tau1 = angular_frequency * tau1
        omega_tau2 = angular_frequency * tau2
        omega_tau3 = angular_frequency * tau3

        bare_capacitance = bare0 * (1 + bare1 * temperature_300) + bare2 * temperature * temperature
        bare_loss = td0 + td1 * temperature + td2 * temperature * temperature
        real1 = populations1 * cls.susceptibility1(temperature, coupling_energy1) / (1.0 + omega_tau1 * omega_tau1) * temperature_curie_inv
        real2 = populations2 * cls.susceptibility2(temperature, coupling_energy2) / (1.0 + omega_tau2 * omega_tau2) * temperature_curie_inv
        real3 = populations3 * cls.susceptibility3(temperature, coupling_energy3) / (1.0 + omega_tau3 * omega_tau3) * temperature_curie_inv
        imag1 = real1 * omega_tau1
        imag2 = real2 * omega_tau2
        imag3 = real3 * omega_tau3

        geometric_factor = bare0 / 4.8

        capacitance = bare_capacitance + geometric_factor * ((emat0 - 1) + emat1 * temperature + np.sum(real1, axis=0) + np.sum(real2, axis=0) + np.sum(real3, axis=0))
        imaginary_capacitance = bare_loss + geometric_factor * (np.sum(imag1, axis=0) + np.sum(imag2, axis=0) + np.sum(imag3, axis=0))
        
        return capacitance, imaginary_capacitance
    
    def show_fit(self):
        fig, axes = self.show_data(film=False)

        capacitance, imaginary_capacitance = self.fitting_function(
            self.temperature_fit, self.angular_frequency, self.curie_temperature,
            self.ln_attempt_time1, self.ln_attempt_time2, self.ln_attempt_time3,
            self.activation_energy1, self.activation_energy2, self.activation_energy3,
            self.coupling_energy1, self.coupling_energy2, self.coupling_energy3,
            self.populations1, self.populations2, self.populations3, self.peak_width1, self.peak_width2, self.peak_width3,
            self.emat0, self.emat1, self.bare0, self.bare1, self.bare2, self.td0, self.td1, self.td2
        )
        # loss = imaginary_capacitance / capacitance

        print(capacitance.shape)
        # print(loss.shape)

        for ii in range(self.freq_num):
            if self.reverse:
                ii = self.freq_num - ii - 1
            axes[0].plot(self.temperature_fit[:, ii], capacitance[:, ii], self.colors_f[ii], linewidth=1)
            axes[1].plot(self.temperature_fit[:, ii], imaginary_capacitance[:, ii], self.colors_f[ii], linewidth=1)
        return fig, axes

    def save(self, filename: str):
        if filename[:-4] not in [".pdf", ".png"]:
            filename += ".png"
        fig, _ = self.show_fit
        fig.savefig(filename, dpi=900, bbox_inches="tight")


class Debye33AAF(Debye33):

    @classmethod
    def susceptibility1(cls, temperature: np.ndarray, coupling_energy: float):
        return cls.susceptibility_antiferroelectric(temperature, coupling_energy)
    
    @classmethod
    def susceptibility2(cls, temperature: np.ndarray, coupling_energy: float):
        return cls.susceptibility_antiferroelectric(temperature, coupling_energy)
    
    @classmethod
    def susceptibility3(cls, temperature: np.ndarray, coupling_energy: float):
        return cls.susceptibility_ferrielectric(temperature, coupling_energy)


class Debye33AAA(Debye33):

    @classmethod
    def susceptibility1(cls, temperature: np.ndarray, coupling_energy: float):
        return cls.susceptibility_antiferroelectric(temperature, coupling_energy)
    
    @classmethod
    def susceptibility2(cls, temperature: np.ndarray, coupling_energy: float):
        return cls.susceptibility_antiferroelectric(temperature, coupling_energy)
    
    @classmethod
    def susceptibility3(cls, temperature: np.ndarray, coupling_energy: float):
        return cls.susceptibility_antiferroelectric(temperature, coupling_energy)
    

class Debye33FFA(Debye33):

    @classmethod
    def susceptibility1(cls, temperature: np.ndarray, coupling_energy: float):
        return cls.susceptibility_ferrielectric(temperature, coupling_energy)
    
    @classmethod
    def susceptibility2(cls, temperature: np.ndarray, coupling_energy: float):
        return cls.susceptibility_ferrielectric(temperature, coupling_energy)
    
    @classmethod
    def susceptibility3(cls, temperature: np.ndarray, coupling_energy: float):
        return cls.susceptibility_antiferroelectric(temperature, coupling_energy)

