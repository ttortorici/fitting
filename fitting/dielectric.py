import numpy as np
from itertools import product
from scipy.optimize import least_squares
import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams["mathtext.fontset"] = "cm"


class Dielectric:
    def __init__(self, temperature, capacitance, loss, frequencies):
        self.temp = temperature
        self.real = capacitance
        self.imag = loss * capacitance
        self.freq = frequencies * 2.0 * np.pi
        
    def initialize_fit(self, num_peaks: int, num_relax_times: int):
        self.num_peaks = num_peaks
        self.num_relax_times = num_relax_times
        self.t = np.array([-29] * num_peaks, dtype=np.float64)
        self.activation_energy = np.arange(num_peaks, dtype=np.float64) * 3.0 + 700.0
        self.peak_center_amplitude = np.ones(num_peaks, dtype=np.float64)
        self.peak_pos_amplitude = np.zeros((int(num_relax_times / 2), num_peaks), dtype=np.float64)
        self.peak_neg_amplitude = np.zeros((int(num_relax_times / 2), num_peaks), dtype=np.float64)

    def fitting_function(self, t, e, s, g, amp, freq, ce300, c1, c2, tdoff, tdlin, tdquad, emat0, emat1):
        """
        
        """

        tau = np.exp()

    def prefactor(self, s):
        """
        s: 2D array of (number of peaks, 1) in fit
        returns: 2D array of (number of peaks, 1) in fit
        """
        sech = 1.0 / np.cosh(s / (2.0 * self.temp))
        return sech * sech

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