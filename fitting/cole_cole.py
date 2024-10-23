from fitting.data import DataSet
from fitting.debye import Debye
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams["mathtext.fontset"] = "cm"


class ColeCole(Debye):
    def __init__(self, data: DataSet, name: str = "", calibration_data: DataSet = None):
        super().__init__(data, name, calibration_data)
        self.emat0 = 3.3
        self.emat1 = 3.7e-4
        self.td0 = 1e-5
        self.td1 = 0.0
        self.td2 = 0.0
        self.num_peaks = 3
        self.curie_temperature = 0.0
        self.ln_attempt_time = np.ones((self.num_peaks, 1, 1), dtype=np.float64) * -30.0
        self.activation_energy = np.array([700.0, 1400.0, 1700.0], dtype=np.float64).reshape((self.num_peaks, 1, 1))
        self.coupling_energy = np.zeros((self.num_peaks, 1, 1), dtype=np.float64)
        self.population = np.ones((self.num_peaks, 1, 1), dtype=np.float64)
        self.cole_term = np.zeros((self.num_peaks, 1, 1), dtype=np.float64)
    
    @classmethod
    def susceptibility1(cls, temperature: np.ndarray, coupling_energy: float):
        return cls.susceptibility_noninteracting(temperature, coupling_energy)
    
    @classmethod
    def susceptibility2(cls, temperature: np.ndarray, coupling_energy: float):
        return cls.susceptibility_noninteracting(temperature, coupling_energy)
    
    @classmethod
    def susceptibility3(cls, temperature: np.ndarray, coupling_energy: float):
        return cls.susceptibility_noninteracting(temperature, coupling_energy)
    
    @classmethod
    def fitting_function(cls, temperature: np.ndarray, angular_frequency, curie_temperature: float,
                         ln_attempt_time: np.ndarray, activation_energy: np.ndarray,
                         coupling_energy: np.ndarray, population: np.ndarray, cole_term: np.ndarray,
                         emat0: float, emat1: float,
                         bare0: float, bare1: float, bare2: float, 
                         td0: float, td1: float, td2: float):
        tau = np.exp(ln_attempt_time + activation_energy / temperature)
        
        temperature_300 = temperature - 300.0
        # temperature_curie_inv = 1. / (temperature - curie_temperature)
        omega_tau = angular_frequency * tau
        omega_tau_cole = omega_tau ** (1-cole_term)
        sin_cole = np.sin(cole_term * 0.5 * np.pi)

        bare_capacitance = bare0 + bare1 * temperature_300 + bare2 * temperature_300 * temperature_300
        bare_loss = td0 + td1 * temperature + td2 * temperature * temperature
        
        susceptibility = cls.susceptibility_antiferroelectric(temperature, coupling_energy)
        # susceptibility = np.stack((cls.susceptibility1(temperature, coupling_energy[0, 0, 0]),
        #                           cls.susceptibility2(temperature, coupling_energy[1, 0, 0]),
        #                           cls.susceptibility3(temperature, coupling_energy[2, 0, 0])))
        denom_inv = population * susceptibility / ((temperature - curie_temperature) * (1.0 + 2.0 * omega_tau_cole * sin_cole + omega_tau_cole * omega_tau_cole))

        real = denom_inv * (1 + omega_tau_cole * sin_cole)
        imag = denom_inv * omega_tau_cole * np.cos(cole_term * 0.5 * np.pi)
        

        geometric_factor = bare0 / 4.8

        capacitance = bare_capacitance + geometric_factor * ((emat0 - 1) + emat1 * temperature + np.sum(real, axis=0))
        imaginary_capacitance = bare_loss + geometric_factor * (np.sum(imag, axis=0))
        
        return capacitance, imaginary_capacitance
    
    def show_fit(self):
        fig, axes = self.show_data(film=False)

        capacitance, imaginary_capacitance = self.fitting_function(
            self.temperature_fit, self.angular_frequency, self.curie_temperature,
            self.ln_attempt_time, self.activation_energy, self.coupling_energy,
            self.population, self.cole_term,
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