import numpy as np
from itertools import product
from pathlib import Path
from scipy.optimize import least_squares
from fitting.dielectric.load import ProcessedFileLite
import matplotlib.pylab as plt

class Debye:
    def __init__(self, file_name_lite, compressor=False):
        self.debye_peaks = 3

        data = ProcessedFileLite(file_name_lite)
        if compressor:
            mask = data.determine_ascending()
            data.data = data.data[mask]

        self.temp = data.get_temperatures()
        self.temp_sq = self.temp * self.temp
        self.temp_shift = self.temp - 300
        self.temp_shift_sq = self.temp_shift * self.temp_shift
        self.real = data.get_real_susceptibilities()
        self.real_error = data.get_real_susceptibility_errors()
        self.imag = data.get_imaginary_susceptibilities()
        self.imag_error = data.get_imaginary_susceptibility_errors()
        self.freqs = np.array(data.freqs)
        self.frequency = self.freqs * 2. * np.pi
        self.freq_num = data.freq_num
        self.data_num = self.temp.shape[0]
        
        self.ln_attempt = -29
        self.activation = np.array([1000, 1700, 2800], dtype=np.float64).reshape((self.debye_peaks, 1, 1))
        self.asymmetry = np.array([0, 40, 190], dtype=np.float64).reshape((self.debye_peaks, 1, 1))
        self.step_size = np.array([80, 120, 150], dtype=np.float64).reshape((self.debye_peaks, 1, 1))
        self.mott = np.ones((self.debye_peaks, 1, 1), dtype=np.float64)
        self.amplitude_c = np.array([.7, 5, 20], dtype=np.float64).reshape((self.debye_peaks, 1, 1))
        self.amplitudes = np.zeros((5, self.debye_peaks, 1, 1), dtype=np.float64)
        self.curie_temperature = 0.

        self.real_bkgd_0 = 5.125
        self.real_bkgd_1 = 2.5e-4
        self.real_bkgd_2 = 0
        self.imag_bkgd_0 = 1e-2
        self.imag_bkgd_1 = 0
        self.imag_bkgd_2 = 0

        params_list = ["ln_attempt", "activation", ]

        self.spacing = np.arange(1, 6, dtype=np.float64).reshape((5, 1, 1, 1))
        self.asym_term = np.empty((self.debye_peaks, self.data_num, self.freq_num))

    def get(self, attribute: str):
        return getattr(self, attribute)
    
    def set(self, attribute: str, value):
        setattr(self, attribute, value)

    # def fit(self, free_params: list[str]):


    def set_upper_temperature(self, upper_temperature: float):
        mask = np.all(self.temp < upper_temperature, axis=1)
        self.temp = self.temp[mask]
        self.temp_sq = self.temp_sq[mask]
        self.temp_shift = self.temp_shift[mask]
        self.temp_shift_sq = self.temp_shift_sq[mask]
        self.real = self.real[mask]
        self.real_error = self.real_error[mask]
        self.imag = self.imag[mask]
        self.imag_error = self.imag_error[mask]
        self.data_num = self.temp.shape[0]
        self.asym_term = np.empty((self.debye_peaks, self.data_num, self.freq_num))
        
    def fitting_function(self, ln_attempt, activation, asymmetry, step_size, mott,
                         amplitude_c, amplitudes, curie_temperature, real_bkgd_0, real_bkgd_1, real_bkgd_2,
                         imag_bkgd_0, imag_bkgd_1, imag_bkgd_2):
        omega_tau_c = self.frequency * np.exp(ln_attempt + (activation / self.temp) ** mott)
        omega_tau_p = self.frequency * np.exp(ln_attempt + ((activation + self.spacing * step_size) / self.temp) ** mott)
        omega_tau_m = self.frequency * np.exp(ln_attempt + ((activation - self.spacing * step_size) / self.temp) ** mott)

        asym_exp = np.exp(1.5 * asymmetry / self.temp)  # shape = (3, d, f)

        self.asym_term[0] = (2. + asym_exp[0]) / (1. + 2. * asym_exp[0])
        self.asym_term[1:] = 3. / (2. + asym_exp[1:])

        omega_tau_c_sq = omega_tau_c * omega_tau_c
        omega_tau_p_sq = omega_tau_p * omega_tau_p
        omega_tau_m_sq = omega_tau_m * omega_tau_m

        real_factor_c = amplitude_c / (1. + omega_tau_c_sq)    # shape = (3, d, f)
        real_factor_p = amplitudes / (1. + omega_tau_p_sq)     # shape = (5, 3, d, f)
        real_factor_m = amplitudes / (1. + omega_tau_m_sq)     # shape = (5, 3, d, f)
        imag_factor_c = real_factor_c * omega_tau_c
        imag_factor_p = real_factor_p * omega_tau_p
        imag_factor_m = real_factor_m * omega_tau_m

        real_factor_o = np.sum(real_factor_p + real_factor_m, axis=0)   # shape = (3, d, f)
        imag_factor_o = np.sum(imag_factor_p + imag_factor_m, axis=0)

        curie = 1. / (self.temp - curie_temperature)    # shape = (d, f)

        real_factor = np.sum((real_factor_c + real_factor_o) * self.asym_term, axis=0) * curie
        imag_factor = np.sum((imag_factor_c + imag_factor_o) * self.asym_term, axis=0) * curie

        real = real_bkgd_0 * (1 + real_bkgd_1 * self.temp_shift + real_bkgd_2 * self.temp_shift_sq) + real_factor
        imag = imag_bkgd_0 + imag_bkgd_1 * self.temp + imag_bkgd_2 * self.temp_sq + imag_factor

        return real, imag
    
    def fitting_for_present(self, temperature, ln_attempt, activation, asymmetry, step_size, mott,
                            amplitude_c, amplitudes, curie_temperature,
                            real_bkgd_0, real_bkgd_1, real_bkgd_2,
                            imag_bkgd_0, imag_bkgd_1, imag_bkgd_2):
        omega_tau_c = self.frequency * np.exp(ln_attempt + (activation / temperature) ** mott)
        omega_tau_p = self.frequency * np.exp(ln_attempt + ((activation + self.spacing * step_size) / temperature) ** mott)
        omega_tau_m = self.frequency * np.exp(ln_attempt + ((activation - self.spacing * step_size) / temperature) ** mott)
        omega_tau_c_sq = omega_tau_c * omega_tau_c
        omega_tau_p_sq = omega_tau_p * omega_tau_p
        omega_tau_m_sq = omega_tau_m * omega_tau_m

        asym_exp = np.exp(1.5 * asymmetry / temperature)  # shape = (3, d, 1)

        self.asym_term[0] = (2. + asym_exp[0]) / (1. + 2. * asym_exp[0])
        self.asym_term[1:] = 3. / (2. + asym_exp[1:])

        real_factor_c = amplitude_c / (1. + omega_tau_c_sq)    # shape = (3, d, f)
        real_factor_p = amplitudes / (1. + omega_tau_p_sq)     # shape = (5, 3, d, f)
        real_factor_m = amplitudes / (1. + omega_tau_m_sq)     # shape = (5, 3, d, f)
        imag_factor_c = real_factor_c * omega_tau_c
        imag_factor_p = real_factor_p * omega_tau_p
        imag_factor_m = real_factor_m * omega_tau_m

        real_factor_o = np.sum(real_factor_p + real_factor_m, axis=0)   # shape = (3, d, f)
        imag_factor_o = np.sum(imag_factor_p + imag_factor_m, axis=0)

        curie = 1. / (temperature - curie_temperature)    # shape = (d, 1)

        real_factor = np.sum((real_factor_c + real_factor_o) * self.asym_term, axis=0) * curie
        imag_factor = np.sum((imag_factor_c + imag_factor_o) * self.asym_term, axis=0) * curie

        real = real_bkgd_0 * (1 + real_bkgd_1 * (temperature - 300) + real_bkgd_2 * (temperature - 300) ** 2) + real_factor
        imag = imag_bkgd_0 + imag_bkgd_1 * temperature + imag_bkgd_2 * temperature * temperature + imag_factor

        return real, imag

    def show_data(self, figsize=None, vertical=True):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if vertical:
            if figsize is None:
                figsize = (6, 8)
            fig, (ax_re, ax_im) = plt.subplots(2, 1, figsize=figsize)
        else:
            if figsize is None:
                figsize = (6.5, 4)
            fig, (ax_re, ax_im) = plt.subplots(1, 2, figsize=figsize)
            ax_re.set_xlabel("Temperature (K)")
        
        for ff, freq in enumerate(self.freqs):
            freq_str = str(int(freq))
            if len(freq_str) > 4:
                freq_str = f"{freq_str[:-3]} kHz"
            else:
                freq_str += " Hz"
            ax_re.scatter(
                self.temp[:, ff],
                self.real[:, ff],
                s=5,
                facecolor="w",
                edgecolor=colors[ff % len(colors)],
                lw=.75,
                label=freq_str,
            )
        ax_im.set_xlabel("Temperature (K)")
        ax_re.set_ylabel("$\\chi'$")
        ax_im.set_ylabel("$\\chi''$")
        for ff, freq in enumerate(self.freqs):
            freq_str = str(int(freq))
            if len(freq_str) > 4:
                freq_str = f"{freq_str[:-3]} kHz"
            else:
                freq_str += " Hz"
            ax_im.scatter(
                self.temp[:, ff],
                self.imag[:, ff],
                s=5,
                facecolor="w",
                edgecolor=colors[ff % len(colors)],
                lw=.75,
                label=freq_str,
            )
        ax_re.grid()
        ax_im.grid()
        ax_re.legend()
        fig.tight_layout()
        return fig, (ax_re, ax_im)
    
    def show_fit(self, figsize=None, vertical=True):
        temperatures = np.linspace(0, self.temp.max(), self.data_num).reshape((self.data_num, 1))
        fig, (ax_re, ax_im) = self.show_data(figsize, vertical)
        real, imag = self.fitting_for_present(temperatures, self.ln_attempt, self.activation, self.asymmetry, self.step_size,
                                              self.mott, self.amplitude_c, self.amplitudes, self.curie_temperature,
                                              self.real_bkgd_0, self.real_bkgd_1, self.real_bkgd_2,
                                              self.imag_bkgd_0, self.imag_bkgd_1, self.imag_bkgd_2)
        for ff, freq in enumerate(self.freqs):
            ax_re.plot(temperatures, real[:, ff])
            ax_im.plot(temperatures, imag[:, ff])
        return fig, (ax_re, ax_im)


if __name__ == "__main__":
    file = Path(r"C:\Users\Teddy\OneDrive - UCB-O365\Rogerslab3\Teddy\TPP Films\BTB-TPP\2024 Film Growth\Film 5 (inclusion 2)\dielectric spectra\2024-09-28__48pBTB-TPP__M07-2__FILM__T-10-35__calibrated_lite.csv")
    fit = Debye(file, compressor=True)
    fit.set_upper_temperature(200)
    fit.show_fit()

    plt.show()
