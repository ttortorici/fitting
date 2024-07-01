import numpy as np
from itertools import product
from scipy.optimize import least_squares
import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams["mathtext.fontset"] = "cm"


class Film:

    eps0 = 8.854187817  # pF/m
    
    def __init__(self, temperature, capacitance, loss, frequencies):
        self.temp = temperature
        self.real = capacitance
        self.imag = loss * capacitance
        self.freq = frequencies.reshape((1, 1, len(frequencies), 1)) * 2.0 * np.pi
        
    def initialize_fit(self, num_peaks: int, num_relax_times: int):
        self.num_peaks = num_peaks
        self.num_relax_times = num_relax_times
        self.relaxation_times = np.arange(num_relax_times, dtype=np.float64).reshape((1, 1, num_relax_times, 1, 1)) * 1.0
        self.t = -30 * np.ones((1, num_peaks, 1, 1), dtype=np.float64)
        self.relaxation_step = np.arange(num_peaks, dtype=np.float64).reshape((1, num_peaks, 1, 1)) * 80.0 + 40.0
        self.relaxation_step[-1] = 2.5e-5
        self.activation_energy = np.arange(num_peaks, dtype=np.float64).reshape((1, num_peaks, 1, 1)) * 3.0 + 700.0
        self.amplitude = np.ones((num_relax_times, num_peaks, 1, 1), dtype=np.float64)
        self.multiple_relaxation = np.arange(self.num_relax_times).reshape((num_relax_times, 1, 1, 1)) - int(self.num_relax_times / 2)
        # self.peak_center_amplitude = np.ones(num_peaks, dtype=np.float64)
        # self.peak_pos_amplitude = np.zeros((int(num_relax_times / 2), num_peaks), dtype=np.float64)
        # self.peak_neg_amplitude = np.zeros((int(num_relax_times / 2), num_peaks), dtype=np.float64)

    def fitting_function(self, t, activation_energy, s, relaxation_step, amp, freq, ce300, c1, c2, tdoff, tdlin, tdquad, emat0, emat1):
        """
        t: shape (1, num_peaks, 1, 1)
        activation_energy: shape (1, num_peaks, 1, 1)
        relaxation_step: shape (1, num_peaks, 1, 1)
        """
        imag_background = tdoff + tdlin * self.temp + tdquad * self.temp * self.temp
        tau = np.exp(t + activation_energy + self.multiple_relaxation * relaxation_step / self.temp)  # shape (num_relax_times, num_peaks, num_freq, data_points)
        chi = np.stack((self.susceptibility_ferri(), self.susceptibility_antiferro()))

    def susceptibility_ferri(self, coupling_energy):
        """
        return: shape (1, 1, num_freq, num_points)
        """
        exp = np.exp(1.5 * coupling_energy / self.temp)
        return (2 + exp) / (1 + 2 * exp)
    
    def susceptibility_antiferro(self, coupling_energy):
        """
        return: shape (1, 1, num_freq, num_points)
        """
        return 3 / (2 + np.exp(1.5 * coupling_energy / self.temp))

    def prefactor(self, s):
        """
        s: 2D array of (number of peaks, 1) in fit
        returns: 2D array of (number of peaks, 1) in fit
        """
        sech = 1.0 / np.cosh(s / (2.0 * self.temp))
        return sech * sech
    
    def show_data(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2,10))
        # colors = ["k", "b", "r"]
        for ii in range(self.freq.shape[2]):
            ax1.scatter(
                self.temp, 
                self.real[:, ii], 
                s=5, 
                edgecolor="k", 
                facecolor="w", 
                marker="o", 
                lw=1, 
                label=f"{int(self.freq[0, 0, ii, 0] / 2.0 / np.pi)}"
            )
            ax2.scatter(
                self.temp, 
                self.imag[:, ii], 
                s=5, 
                edgecolor="k", 
                facecolor="w",
                marker="o", 
                lw=1, 
                label=f"{int(self.freq[0, 0, ii, 0] / 2.0 / np.pi)}"
            )
        # ax1.set_xlabel("Temperature (K)")
        ax1.set_ylabel("Real Capacitance (pF)")
        ax1.legend(title="Frequency (Hz)")
        # ax1.set_ylim(range1[0], range1[1])
        # ax1.set_xlim(0, 250)
        # ax1.set_title(name1)
        ax2.set_xlabel("Temperature (K)")
        ax2.set_ylabel("Imaginary Capacitance (pF)")
        # ax3.set_xlim(0, 250)
        # ax3.set_ylim(range1[2], range1[3])
        ax1.grid()
        ax2.grid()
        # ax3.grid()
        # ax4.grid()
        ax1.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax2.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax1.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        ax2.tick_params(axis="both", which="both", direction="in", top=True, right=True)

        plt.tight_layout()

    @staticmethod
    def convert_energy(energy, units):
        if units.lower() == "kcal":
            kB = 0.001987          # energy in kcal/mol
        elif units.lower() == "k":
            kB = 1.0               # energy in K
        elif units.lower() == "j":
            kB = 1.38064852e-23    # energy in J
        elif units.lower() == "ev":
            kB = 0.000086173303    # energy in eV
        else:
            raise ValueError("Energy units not recognized.")
        return energy * kB
    
    @staticmethod
    def modulus(gap, unit_cell, film_thickness):
        pi_over_4h = np.pi / (4.0 * film_thickness)
        three_u_minus_g = 3.0 * unit_cell - gap
        x = np.sinh(pi_over_4h * three_u_minus_g) / np.sinh(pi_over_4h * (unit_cell + gap))
        y = np.sinh(pi_over_4h * three_u_minus_g) / np.sinh(pi_over_4h * (unit_cell - gap))
        return (x - 1) * (x + 1) / ((y - 1) * (y + 1))

    @staticmethod
    def geometric_factor(gap, unit_cell, finger_length, finger_num):
        return (finger_num - 1) * finger_length * np.pi / np.log(16.0 / Film.modulus(gap, unit_cell, finger_length))