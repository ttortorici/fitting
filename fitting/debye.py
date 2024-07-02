from data import DataSet
from bare import Bare
import numpy as np
from itertools import product
from scipy.optimize import least_squares
import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams["mathtext.fontset"] = "cm"


class Debye:

    eps0 = 0.008854187817  # pF/mm

    def __init__(self, data: DataSet, name: str="", calibration_data: DataSet=None):
        self.freq_num = data.freq_num
        self.name = name
        if self.freq_num == 3:
            self.colors_d = ["k", "b", "r"]
            self.colors_f = ["r", "r", "k"]
        else:
            self.colors_d = ["k", "darkgreen", "turquoise", "b", "slateblue", "darkviolet", "r"]
            self.colors_f = ["r", "r", "r", "r", "r", "r", "k"]
        
        self.reverse = data.reverse
        if self.reverse:
            self.colors_d = self.colors_d[::-1]
            self.colors_f = self.colors_f[::-1]

        self.frequencies = data.frequencies
        self.angular_frequency = np.array(self.frequencies) * 2 * np.pi
        
        if calibration_data is not None:
            calibration = Bare(calibration_data)
            calibration.fit()
            calibration.show()
            self.bare0 = calibration.c300       # bare capacitance at 300 K
            self.bare1 = calibration.linear     # linear term of bare capacitance
            self.bare2 = calibration.quadratic  # quadratic term of bare capacitance
        else:
            self.bare0 = np.ones(self.freq_num) * 1.4
            self.bare1 = 1.8e-5
            self.bare2 = 0.0
        
        self.temperature = data.get_temperatures()
        self.capacitance = data.get_capacitances()
        self.losstangent = data.get_losses()
        
        self.temperature_fit = np.stack(
            [np.linspace(self.temperature.min()-10, self.temperature.max()+10, 1000)] * self.freq_num
        ).T
        
    @staticmethod
    def susceptibility_noninteracting(temperature: np.ndarray, coupling_energy: float):
        sech = 1.0 / np.cosh(coupling_energy / (2.0 * temperature))
        return sech * sech
    
    @staticmethod
    def susceptibility_antiferroelectric(temperature: np.ndarray, coupling_energy: float):
        return 3 / (2 + np.exp(1.5 * coupling_energy / temperature))
    
    @staticmethod
    def susceptibility_ferrielectric(temperature: np.ndarray, coupling_energy: float):
        exp = np.exp(1.5 * coupling_energy / temperature)
        return (2 + exp) / (1 + 2 * exp)

    def show_data(self, film=True):
        
        fig, axes = plt.subplots(2, 1, figsize=(4, 9))
        axes[0].set_title(self.name)
        for ax in axes:
            ax.grid(linestyle='dotted')
            ax.set_xlabel('Temperature (K)')
            ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
            ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        
        if film:
            axes[0].set_ylabel("$\\chi'$")
            axes[1].set_ylabel("$\\chi''$")
        else:
            axes[0].set_ylabel("Capacitance (pF)")
            axes[1].set_ylabel("$\\tan\\delta$")

        for ii in range(self.freq_num):
            if self.reverse:
                ii = self.freq_num - ii - 1
            freq_name = str(int(self.frequencies[ii]))
            if len(freq_name) > 3:
                freq_name = freq_name[:-3] + " kHz"
            else:
                freq_name += " Hz"
            ax[0].scatter(self.data[:, ii], self.data[:, ii], s=4, marker="o",
                          edgecolors=self.colors[ii], lw=.75, alpha=1, facecolor='w',
                          label=freq_name)
            ax[1].scatter(self.data[:, ii], self.data[:, ii], s=4, marker="o",
                          edgecolors=self.colors[ii], lw=.75, alpha=1, facecolor='w')

        fig.tight_layout()
        return fig, axes


class Debye33(Debye):

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
        self.populations1[5, 1, 1] = 1.0        # 5 is the center
        self.populations2[5, 1, 1] = 1.0        # 5 is the center
        self.populations3 = 1.0
        self.peak_width1 = 50.0
        self.peak_width2 = 50.0
    
    @classmethod
    def fitting_function(cls, temperature: np.ndarray, angular_frequency, curie_temperature: float, ln_attempt_time: float,
                         activation_energy1: float, activation_energy2: float, activation_energy3: float,
                         coupling_energy1: float, coupling_energy2: float, coupling_energy3: float,
                         populations1: np.ndarray, populations2: np.ndarray, populations3: float,
                         peak_width1: float, peak_width2: float, emat0: float, emat1: float,
                         bare0: float, bare1: float, bare2: float, 
                         td0: float, td1: float, td2: float):
        offsets = np.linspace(-5, 5, 11).reshape((11, 1, 1))
        tau1 = np.exp(ln_attempt_time + (activation_energy1 + offsets * peak_width1) / temperature)     # shape (num_relax_times, 1, 1)
        tau2 = np.exp(ln_attempt_time + (activation_energy2 + offsets * peak_width2) / temperature)     # shape (num_relax_times, 1, 1)
        tau3 = np.exp(ln_attempt_time + (activation_energy3) / temperature)

        temperature_300 = temperature - 300.0
        temperature_curie_inv = 1. / (temperature - curie_temperature)
        omega_tau1 = angular_frequency * tau1
        omega_tau2 = angular_frequency * tau2
        omega_tau3 = angular_frequency * tau3

        bare_capacitance = bare0 + bare1 * temperature_300 + bare2 * temperature_300 * temperature_300
        bare_loss = td0 + td1 * temperature + td2 * temperature * temperature
        real1 = populations1 * cls.susceptibility_noninteracting(temperature, coupling_energy1) / (1.0 + omega_tau1 * omega_tau1) * temperature_curie_inv
        real2 = populations2 * cls.susceptibility_noninteracting(temperature, coupling_energy2) / (1.0 + omega_tau2 * omega_tau2) * temperature_curie_inv
        real3 = populations3 * cls.susceptibility_noninteracting(temperature, coupling_energy3) / (1.0 + omega_tau3 * omega_tau3) * temperature_curie_inv
        imag1 = real1 * omega_tau1
        imag2 = real2 * omega_tau2
        imag3 = real3 * omega_tau3

        geometric_factor = bare0 / 4.8

        capacitance = bare_capacitance + geometric_factor * (emat0 - 1) + emat1 * temperature_300 + np.sum(real1, axis=0) + np.sum(real2, axis=0) + real3
        loss = bare_loss + geometric_factor * (np.sum(imag1, axis=0) + np.sum(imag2, axis=0) + imag3)
        
        return capacitance, loss
    
    
    


class Film:

    eps0 = 0.008854187817  # pF/mm
    colors_d = ["k", "darkgreen", "turquoise", "b", "slateblue", "darkviolet", "r"]
    colors_f = ["r", "r", "r", "r", "k", "k", "k"]

    def __init__(self, temperature, capacitance, loss, frequencies):
        imag = loss * capacitance      # shape (num_points, num_freq)
        if type(frequencies) is not np.ndarray:
            frequencies = np.array(frequencies)
        self.freq = frequencies * 2.0 * np.pi
        
        self.real = np.zeros(capacitance.shape)
        self.imag = np.zeros(imag.shape)
        self.temperature = np.zeros(temperature.shape)

        for ii in range(len(frequencies)):
            sort_ind = np.argsort(temperature[:, ii])
            self.real[:, ii] = capacitance[sort_ind, ii]
            self.imag[:, ii] = imag[sort_ind, ii]
            self.temperature[:, ii] = temperature[sort_ind, ii]

    def initialize_fit(self, capacitor_gap, film_thickness):
        self.coupling_energy = 10.0
        self.attempt_time = -30     # tau_0 = exp(-30)
        self.curie_temperature = 0.0
        self.activation_energy = 700.0
        self.amplitude = np.ones((3, 1, 1), dtype=np.float64)
        self.real_b_300 = 3.9
        self.real_b_1 = 0.0
        self.real_b_2 = 0.0
        self.im_b_0 = 0.0
        self.im_b_1 = 0.0
        self.im_b_2 = 0.0
        self.gap_width = capacitor_gap
        self.unit_width = 20.0
        self.substrate_thickness = 500.0
        self.num_fingers = 50
        self.finger_length = 1e-3
        self.finger_contribution = self.eps0 * (self.num_fingers - 1) * self.finger_length
        self.film_modulus = self.modulus(capacitor_gap, self.unit_width, film_thickness)
        self.film_geometric = (self.num_fingers - 1) * self.finger_length * np.pi / np.log(16.0 / self.film_modulus)

    def fitting_function_AAF(self, curie_temperature, attempt_time, activation_energy, coupling_energy, amplitude, tunneling,
                            real_b_300, real_b_1, real_b_2, im_b_0, im_b_1, im_b_2):
        """
        :param curie_temperature: The temperature that the dielectric constant blows up to infinity at
        :param attempt_time: natural log of the attempt time in seconds (should be negative... corresponding to fast)
        :param activation_energy: barriers to rotation in Kelvin (500 K = 1 kCal/mol).. there should be 3 of these, 1 for each peak.
        :param coupling_energy: intra-channel dipole coupling.. there should be 3 of these, 1 for each peak.
        :param amplitude: amplitude of each of the 3 peaks
        :param real_b_300: background capacitance at room temperature
        :param real_b_1: linear term of background capacitance
        :param real_b_2: quadratic term of background capacitance
        :param im_b_0: offset term of loss
        :param im_b_1: linear term of loss
        :param im_b_2: quadratic term of loss
        """
        temp_300 = self.temperature - 300.0
        tau = np.exp(attempt_time + (activation_energy / self.temperature) ** tunneling)    # (data_points, frequencies)
        exp = np.exp(1.5 * coupling_energy / self.temperature)
        anti = 3 / (2 + np.exp(1.5 * coupling_energy / self.temperature))
        ferri = (2 + exp) / (1 + 2 * exp)

        chi = np.stack((ferri, anti, anti))      # (3, data_points, frequencies)
        omega_tau = self.freq * tau     # (data_points, frequencies)
        real = np.sum(amplitude / np.abs(self.temperature - curie_temperature) * chi / (1.0 + omega_tau * omega_tau), axis=0)
        imag = real * omega_tau
        return real, imag

    
    def antiferroelectric_susceptibility(self, coupling_energy):
        return 3 / (2 + np.exp(1.5 * coupling_energy / self.temperature))
    
    def ferrielectric_susceptibility(self, coupling_energy):
        exp = np.exp(1.5 * coupling_energy / self.temperature)
        return (2 + exp) / (1 + 2 * exp)

    def fitting_function(self, tc, t, activation_energy, s, relaxation_step, amp, real_b_300, real_b_1, real_b_2, im_b_0, im_b_1, im_b_2, gap_width, unit_width, substrate_thickness):
        
        modulus_sq = 2.0 * (unit_width - gap_width) ** 3 / ((unit_width + gap_width) ** 2 * (2 * unit_width - gap_width))
        
        root_kp = (1 - modulus_sq) ** 0.25
        root_kp = (1 - modulus_sq) ** 0.25
        ellint_ratio = np.pi / np.log(2 * (1 + root_kp) / (1 - root_kp))

        temp_300 = self.temperature - 300.0
        tau = np.exp(t + activation_energy + self.multiple_relaxation * relaxation_step / self.temperature)  # shape (num_relax_times, data_points, num_freq)
        chi = self.susceptibility(s)
        omega_tau = self.freq * tau
        dielectric_real = amp / (self.temperature - tc) * chi / (1.0 + omega_tau * omega_tau)
        dielectric_imag = dielectric_real * omega_tau
        dielectric_real = np.sum(dielectric_real, axis=0)
        dielectric_imag = np.sum(dielectric_imag, axis=0)

        real = real_b_300 + real_b_1 * temp_300 + real_b_2 * temp_300 * temp_300 + dielectric_real
        imag = im_b_0 + im_b_1 * self.temperature + im_b_2 * self.temperature * self.temperature + dielectric_imag

        real_capacitance = self.finger_contribution * ellint_ratio * (1 + real)
        imag_capacitance = self.finger_contribution * imag * ellint_ratio
        return real_capacitance, imag_capacitance

    def susceptibility(self, s):
        """
        s: float
        returns: 2D array of (num_freq, num_points)
        """
        sech = 1.0 / np.cosh(s / (2.0 * self.temperature))
        return sech * sech
    
    def show_data(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.2,10))
        for ii in range(len(self.freq)):
            ax1.scatter(
                self.temperature[:, ii], 
                self.real[:, ii], 
                s=10, 
                edgecolor=self.colors_d[ii], 
                facecolor=None, 
                marker="o", 
                lw=.5, 
                alpha=.25,
                label=f"{int(np.round(self.freq[ii] / 2.0 / np.pi, -2))}"
            )
            ax2.scatter(
                self.temperature[:, ii], 
                self.imag[:, ii], 
                s=10, 
                edgecolor=self.colors_d[ii], 
                facecolor=None,
                marker="o", 
                lw=.5, 
                alpha=.25,
                label=f"{int(np.round(self.freq[ii] / 2.0 / np.pi, -2))}"
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

    def show_fit(self):
        result = self.fitting_function(self.tc, self.t, self.activation_energy,
                                       self.s, self.relaxation_step, self.amplitude,
                                       self.real_b_300, self.real_b_1, self.real_b_2,
                                       self.im_b_0, self.im_b_1, self.im_b_2, self.gap_width,
                                       self.unit_width, self.substrate_thickness)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.2,10))
        for ii in range(len(self.freq)):
            ax1.scatter(
                self.temperature[:, ii], 
                self.real[:, ii], 
                s=10, 
                edgecolor=self.colors_d[ii], 
                facecolor=None, 
                marker="o", 
                lw=.5, 
                alpha=.25,
                label=f"{int(np.round(self.freq[ii] / 2.0 / np.pi, -2))}"
            )
            ax1.plot(self.temperature[:, ii], result[0][:, ii], color=self.colors_f[ii], lw=1.5)
            ax2.scatter(
                self.temperature[:, ii], 
                self.imag[:, ii], 
                s=10, 
                edgecolor=self.colors_d[ii], 
                facecolor=None,
                marker="o", 
                lw=.5, 
                alpha=.25,
                label=f"{int(np.round(self.freq[ii] / 2.0 / np.pi, -2))}"
            )
            ax2.plot(self.temperature[:, ii], result[1][:, ii], color=self.colors_f[ii], lw=1.5)
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

    @classmethod
    def geometric_factor(cls, gap, unit_cell, thickness, finger_length, finger_num):
        return (finger_num - 1) * finger_length * np.pi / np.log(16.0 / cls.modulus(gap, unit_cell, thickness))
    
