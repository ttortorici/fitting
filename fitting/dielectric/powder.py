from scipy.optimize import least_squares
from scipy.special import erf
from pathlib import Path
from fitting.dielectric.data import RawData, Unsorted, ProcessedPowder
from fitting.dielectric.variance import calculate as calc_variance
import numpy as np
import matplotlib.pylab as plt
plt.style.use("fitting.style")


root2pi_inv = 1. / np.sqrt(2.0 * np.pi)

subscript = ["\u2080", "\u2081", "\u2082", "\u2083", "\u2084",
             "\u2085", "\u2086", "\u2087", "\u2088", "\u2089"]
superscript = ["\u2070", "\u00B9", "\u00B2", "\u00B3", "\u2074",
               "\u2075", "\u2076", "\u2077", "\u2078", "\u2079"]


class Powder:
    def __init__(self, data_files: list[Path],
                 room_temperature_capacitance: float,
                 linear_term: float,
                 quadratic_term: float, 
                 quartic_term: float,
                 epsilon_substrate: float, already_sorted=False):
        print("Loading Powder data. Loading data from file(s):")
        self.sorted = already_sorted
        if isinstance(data_files, str):
            data_files = Path(data_files)
        if isinstance(data_files, Path):
            data_files = [data_files,]
            self.dir = data_files.parent
            self.name = data_files.stem
        else:
            self.dir = data_files[0].parent
            self.name = data_files[0].stem
        for file in data_files:
            if isinstance(data_files, str):
                print(f' - "{file}"')
            else:
                print(f' - "{file.as_posix()}"')
        self.time = None
        self.temp = None
        self.caps = None
        self.loss = None
        self.freq = None
        self.fnum = None
        self.real_order = None
        self.imag_order = None
        self._fit_real = None
        self._fit_imag = None
        self._dof_real = None
        self._dof_imag = None
        self._red_chi_sq_real = None
        self._red_chi_sq_imag = None
        self.standard_dev_real = None
        self.standard_dev_imag = None
        self.ascending = None
        self.data = None
        self.inv_eps0_G0 = (1 + epsilon_substrate) / room_temperature_capacitance

        self.load_file(data_files)

        if room_temperature_capacitance is None:
            self.bare = None
        else:
            temperatures_300 = self.temp - 300.
            temp_300_sq = temperatures_300 * temperatures_300
            self.bare = np.ones_like(temperatures_300) * room_temperature_capacitance * (1
                            + linear_term * temperatures_300
                            + quadratic_term * temp_300_sq
                            + quartic_term * temp_300_sq * temperatures_300)
            self.real_shift = self.caps - self.bare
            self.imag_shift = self.caps * self.loss
            self.std_shift_real, self.std_shift_imag = calc_variance(self.temp, self.real_shift, self.imag_shift, 5)

            self.eps_real = 1 + self.real_shift * self.inv_eps0_G0
            self.eps_imag = self.imag_shift * self.inv_eps0_G0
            self.std_eps_real, self.std_eps_imag = calc_variance(self.temp, self.eps_real, self.eps_imag, 5)

        # self.std_real, self.std_imag = self.determine_variance(5, 1)

    def run(self, max_temperature_data: float=None):
        if max_temperature_data is not None:
            temperature_mask = np.all(self.temp < max_temperature_data, axis=1)
        else:
            temperature_mask = np.ones(self.temp.shape[0], dtype=bool)

        data_pts = len(self.time)       # number of rows
        print(f"num data: {data_pts}")

        data_at_freq = [None] * self.fnum
        labels = []
        labels_lite = []
        for ff, freq in enumerate(self.freq):
            labels.extend([f"{lab} ({int(freq)} Hz)" for lab in ProcessedPowder.LABELS])
            data_at_freq[ff] = np.stack((
                self.time[:, ff][temperature_mask],
                self.temp[:, ff][temperature_mask],
                self.caps[:, ff][temperature_mask],                 # C'
                self.loss[:, ff][temperature_mask],                 # loss
                self.real_shift[:, ff][temperature_mask],           # del C'
                self.std_shift_real[:, ff][temperature_mask],
                self.eps_real[:, ff][temperature_mask],             # eps' effective
                self.std_eps_real[:, ff][temperature_mask],
                self.imag_shift[:, ff][temperature_mask],           # del C''
                self.std_shift_imag[:, ff][temperature_mask],
                self.eps_imag[:, ff][temperature_mask],             # eps'' effective
                self.std_eps_imag[:, ff][temperature_mask],
                np.ones(data_pts) * freq,
            ), axis=1)
        data = np.hstack(data_at_freq)
        labels = ", ".join(labels)
        labels_lite = ", ".join(labels_lite)
        np.savetxt(self.dir / (self.name + "__powder-process.csv"), data, delimiter = ", ", header=labels)

    def determine_variance_pt2pt(self, slice_size: int):
        if self.bare is not None:
            real = self.real_shift
        else:
            real = self.caps
        imaginary = self.imag_shift
        slices = np.arange(0, len(self.time), slice_size)
        slices[-1] = len(self.time)

        std_devs_re = np.empty_like(self.time)
        std_devs_im = np.empty_like(self.time)
        for ff in range(self.fnum):
            time = self.time[:, ff]
            capacitance_shift = real[:, ff]
            imaginary_capacitance = imaginary[:, ff]
            for ss in range(len(slices) - 1):
                real_slice = capacitance_shift[slices[ss] : slices[ss + 1]]
                imag_slice = imaginary_capacitance[slices[ss] : slices[ss + 1]]
                pt2pt_diff_re = real_slice[1:] - real_slice[:1]
                pt2pt_diff_im = imag_slice[1:] - imag_slice[:1]

                variance_c = np.sum(pt2pt_diff_re * pt2pt_diff_re) / (len(real_slice) - 1)
                variance_l = np.sum(pt2pt_diff_im * pt2pt_diff_im) / (len(imag_slice) - 1)
                std_devs_re[slices[ss]: slices[ss + 1], ff] = np.sqrt(variance_c)
                std_devs_im[slices[ss]: slices[ss + 1], ff] = np.sqrt(variance_l)
        return std_devs_re, std_devs_im


    def determine_variance(self, slice_size: int, poly_order: int):
        params_re = [0] * (poly_order + 1)
        params_im = params_re.copy()

        if self.bare is not None:
            real = self.real_shift
        else:
            real = self.caps
        imaginary = self.imag_shift
        slices = np.arange(0, len(self.time), slice_size)
        if len(self.time) - slices[-1] > poly_order:
            slices = np.append(slices, len(self.time))
        else:
            slices[-1] = len(self.time)
        
        std_devs_re = np.empty_like(self.time)
        std_devs_im = np.empty_like(self.time)
        for ff in range(self.fnum):
            temp = self.temp[:, ff]
            capacitance_shift = real[:, ff]
            imaginary_capacitance = imaginary[:, ff]
            for ss in range(len(slices) - 1):
                temp_slice = temp[slices[ss] : slices[ss + 1]]
                capt_slice = capacitance_shift[slices[ss] : slices[ss + 1]]
                loss_slice = imaginary_capacitance[slices[ss] : slices[ss + 1]]
                params_re[0] = np.average(capt_slice)
                params_im[0] = np.average(loss_slice)
                fit_capt = least_squares(RawData.residuals, params_re, args=(temp_slice, capt_slice), method="lm")
                fit_loss = least_squares(RawData.residuals, params_im, args=(temp_slice, loss_slice), method="lm")
                curve_slice_c = RawData.poly_fit(fit_capt.x, temp_slice)
                curve_slice_l = RawData.poly_fit(fit_loss.x, temp_slice)
                diff_c = curve_slice_c - capt_slice
                diff_l = curve_slice_l - loss_slice

                variance_c = np.sum(diff_c * diff_c) / (len(capt_slice) - 1)
                variance_l = np.sum(diff_l * diff_l) / (len(loss_slice) - 1)
                std_devs_re[slices[ss]: slices[ss + 1], ff] = np.sqrt(variance_c)
                std_devs_im[slices[ss]: slices[ss + 1], ff] = np.sqrt(variance_l)
        return std_devs_re, std_devs_im

    def load_file(self, file: Path):
        if self.sorted:
            data = RawData(file)
        else:
            data = Unsorted(file)
        self.time = data.get_times()
        self.temp = data.get_temperatures()
        self.caps = data.get_capacitances()
        self.loss = data.get_losses()
        self.freq = data.freqs
        self.fnum = data.freq_num


    def variance_point_to_point(self):
        # derivative = (self.temp[1:] - self.temp[:-1]) / (self.time[1:] - self.time[:-1])
        point_to_point_difference = (self.caps[1:] - self.caps[:-1]) # / np.exp(derivative)
        variance = np.sum(point_to_point_difference * point_to_point_difference,axis=0) / (self.caps.shape[0] - 1)
        standard_deviation = np.sqrt(variance)
        print(f"Cap Pt-Pt    ", end="")
        for std in standard_deviation:
            string = f"{std:.3e}"
            string = " |" + " " * (11 - len(string)) + string
            print(string, end="")
        print(" |")

        point_to_point_difference = (self.loss[1:] - self.loss[:-1]) # / derivative
        variance = np.sum(point_to_point_difference * point_to_point_difference,axis=0) / (self.caps.shape[0] - 1)
        standard_deviation = np.sqrt(variance)
        print(f"Loss Pt-Pt   ", end="")
        for std in standard_deviation:
            string = f"{std:.3e}"
            string = " |" + " " * (11 - len(string)) + string
            print(string, end="")
        print(" |")
        

    def show_fit(self, peaks: bool):
        prop_cycle = plt.rcParams['axes.prop_cycle']
        color = prop_cycle.by_key()['color']
        fig, axes = plt.subplots(2, 1, figsize=(6, 8))
        x = np.linspace(4, 400, 10000)
        x = np.dstack([x] * self.temp.shape[1])[0]

        fit_to_plot = self.fit_function_real(self._fit_real, x)
        if fit_to_plot.ndim == 1:
            fit_to_plot = np.tile(fit_to_plot, (x.shape[0], 1))
        for ii in range(self.fnum):
            axes[0].scatter(self.temp[:, ii], self.caps[:, ii],
                            s=3, edgecolor=color[ii], facecolor='w', label=f"{int(self.freq[ii])} Hz")
            axes[0].plot(x, fit_to_plot[:, ii], color[ii])
        
        if peaks:
            fit_to_plot = self.fit_function_imag_peaks(self._fit_imag, x)
        else:
            fit_to_plot = self.fit_function_imag_poly(self._fit_imag, x)
            if fit_to_plot.shape[0] == 1:
                fit_to_plot = np.tile(fit_to_plot, (x.shape[0], 1))
        for ii in range(self.fnum):
            axes[1].scatter(self.temp[:, ii], self.loss[:, ii],
                            s=3, edgecolor=color[ii], facecolor='w', label=f"{int(self.freq[ii])} Hz")
            axes[1].plot(x, fit_to_plot[:, ii], color[ii])

        for ax in axes:
            ax.grid()
            ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        axes[0].set_ylabel('Capacitance (pF)')
        axes[1].set_ylabel('Loss Tangent')
        axes[1].set_xlabel('Temperature (K)')

        real_text = f"polynomial order = {self.real_order}"
        imag_text = f"polynomial order = {self.imag_order}"
        axes[0].text(400, self.caps.max() * 0.1 + self.caps.min() * 0.9, real_text, ha="right")
        axes[1].text(0.9, 0.9, imag_text, ha="right", transform=ax.transAxes)
        axes[0].legend()
        axes[0].set_title("Bare Capacitance Fit")
        fig.tight_layout()





if __name__ == "__main__":
    import matplotlib.pylab as plt

    file = Path(r"H:\OneDrive - UCB-O365\Rogerslab3\Teddy\Thesis\chapter-4\Data\BDS\1@TPP sat - GBA 124\Cooling_1468959484_96 - Copy.csv")
    pow = Powder(file, 1.175)
    