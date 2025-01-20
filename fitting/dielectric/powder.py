from scipy.optimize import least_squares
from scipy.special import erf
from pathlib import Path
from fitting.dielectric.load import RawData, Unsorted, ProcessedPowder
import numpy as np
import matplotlib.pylab as plt
plt.style.use("fitting.style")


root2pi_inv = 1. / np.sqrt(2.0 * np.pi)

subscript = ["\u2080", "\u2081", "\u2082", "\u2083", "\u2084",
             "\u2085", "\u2086", "\u2087", "\u2088", "\u2089"]
superscript = ["\u2070", "\u00B9", "\u00B2", "\u00B3", "\u2074",
               "\u2075", "\u2076", "\u2077", "\u2078", "\u2079"]


class Powder:
    def __init__(self, data_files: list[Path], room_temperature_capacitance: float = None, linear_term: float = 2.3e-5, quadratic_term: float = 3e-8, sorted=False):
        print("Loading Powder data. Loading data from file(s):")
        self.sorted = sorted
        if isinstance(data_files, str):
            data_files = Path(data_files)
        if isinstance(data_files, Path):
            data_files = [data_files,]
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

        self.load_file(data_files)

        if room_temperature_capacitance is None:
            self.bare = None
        else:
            temperatures_300 = self.temp - 300.
            self.bare = np.ones_like(temperatures_300) * room_temperature_capacitance * (1
                            + linear_term * temperatures_300 + quadratic_term * temperatures_300 * temperatures_300)
        self.std_real, self.std_imag = self.determine_variance(10, 1)
        print(self.std_real)
        self.std_real_p2p, self.std_imag_p2p = self.determine_variance_pt2pt(10)
        print(self.std_real_p2p)
        print(self.std_real_p2p / self.std_real)
        self.variance_point_to_point()
        fig, ax = plt.subplots(2, 1, figsize=(6, 8))
        for ii in range(3):
            ax[0].plot(self.temp[:, ii], self.std_real[:, ii], label=f"{self.freq[ii]}")
            ax[0].plot(self.temp[:, ii], self.std_real_p2p[:, ii], label=f"{self.freq[ii]} p2p")
            ax[1].plot(self.temp[:, ii], self.std_imag[:, ii], label=f"{self.freq[ii]}")
            ax[1].plot(self.temp[:, ii], self.std_imag_p2p[:, ii], label=f"{self.freq[ii]} p2p")
        ax[0].set_title("Real Error")
        ax[1].set_title("Imaginary Error")
        for ii in range(2):
            ax[ii].set_xlabel("Temperature (K)")
            ax[ii].legend()
            ax[ii].grid()

    def run(self, max_temperature_data: float=None):
        if max_temperature_data is not None:
            temperature_mask = np.all(self.raw_data.get_temperatures() < max_temperature_data, axis=1)
        else:
            temperature_mask = np.ones(self.raw_data.shape[0], dtype=bool)

        data_pts = self.raw_data.shape[0]       # number of rows
        print(f"num data: {data_pts}")
        time = self.raw_data.get_times()
        temp = self.raw_data.get_temperatures()
        tempb = self.raw_data.get_shield_temperatures()
        caps = self.raw_data.get_capacitances()

        loss = self.raw_data.get_losses()
        volt = self.raw_data.get_voltages()
        cap_imag = caps * loss
        cap_bare_imag = self.bare_cap_curve * self.bare_loss_curve
        cap_err_imag = np.sqrt((self.raw_data.cap_std * loss)**2 + (self.raw_data.loss_std * caps)**2)
        cap_err_bare_imag = np.sqrt((self.bare_cap_dev * self.bare_loss_curve)**2 + (self.bare_loss_dev * self.bare_cap_curve)**2)
        del_cap_err_real = np.sqrt(self.raw_data.cap_std * self.raw_data.cap_std + self.bare_cap_dev * self.bare_cap_dev)
        del_cap_err_imag = np.sqrt(cap_err_imag * cap_err_imag + cap_err_bare_imag * cap_err_bare_imag)
        
        del_cap_real = (caps - self.bare_cap_curve)     # pF
        del_cap_imag = (cap_imag - cap_bare_imag)       # pF

        (eps_real, eps_imag), (eps_real_err, eps_imag_err) = capacitor.dielectric_constant(
            del_cap_real, del_cap_imag, gap_width, film_thickness, finger_length=1e-3,
            finger_num=finger_num, delta_cap_real_err=del_cap_err_real, delta_cap_imag_err=del_cap_err_imag,
            gap_err=gap_err, film_thickness_err=film_thickness_err, finger_length_err=finger_length_err
        )

        eps_real_std, eps_imag_std = self.raw_data.determine_variance(15, 1, eps_real, eps_imag)

        data_at_freq = [None] * self.raw_data.freq_num
        data_at_freq_lite = [None] * self.raw_data.freq_num
        labels = []
        labels_lite = []
        for ff, freq in enumerate(self.raw_data.freqs):
            labels.extend([f"{lab} ({int(freq)} Hz)" for lab in ProcessedFile.LABELS])
            labels_lite.extend([f"{lab} ({int(freq)} Hz)" for lab in ProcessedFileLite.LABELS])
            data_at_freq[ff] = np.stack((
                time[:, ff],
                temp[:, ff],
                tempb[:, ff],
                caps[:, ff],                                # C'
                self.raw_data.cap_std[:, ff],
                self.bare_cap_curve[:, ff],                 # C'_b
                np.ones(data_pts) * self.bare_cap_dev[ff],
                del_cap_real[:, ff],                        # del C'          
                del_cap_err_real[:, ff],
                loss[:, ff],                                # tan(delta)
                self.raw_data.loss_std[:, ff],
                self.bare_loss_curve[:, ff],                # tan(delta)_b
                np.ones(data_pts) * self.bare_loss_dev[ff],
                cap_imag[:, ff],                            # C''
                cap_err_imag[:, ff],
                cap_bare_imag[:, ff],                       # C''_b
                cap_err_bare_imag[:, ff],
                del_cap_imag[:, ff],                        # del C''
                del_cap_err_imag[:, ff],
                eps_real[:, ff],                            # real dielectric constant
                eps_real_std[:, ff],
                eps_real_err[:, ff],
                eps_imag[:, ff],                            # imaginary dielectric constant
                eps_imag_std[:, ff],
                eps_imag_err[:, ff],
                volt[:, ff],
                np.ones(data_pts) * freq,
            ), axis=1)
            data_at_freq_lite[ff] = np.stack((
                time[temperature_mask][:, ff],
                temp[temperature_mask][:, ff],
                caps[temperature_mask][:, ff] - self.bare_cap_curve[temperature_mask][:, ff],   # del C'          
                del_cap_err_real[temperature_mask][:, ff],
                del_cap_imag[temperature_mask][:, ff],                        # del C''
                del_cap_err_imag[temperature_mask][:, ff],
                eps_real[temperature_mask][:, ff],                            # real dielectric constant
                eps_real_std[temperature_mask][:, ff],
                eps_imag[temperature_mask][:, ff],                            # imaginary dielectric constant
                eps_imag_std[temperature_mask][:, ff],
                volt[temperature_mask][:, ff],
                np.ones(np.sum(temperature_mask)) * freq,
            ), axis=1)
        data = np.hstack(data_at_freq)
        data_lite = np.hstack(data_at_freq_lite)
        labels = ", ".join(labels)
        labels_lite = ", ".join(labels_lite)
        np.savetxt(self.dir / (self.name + "__calibrated.csv"), data, delimiter = ", ", header=labels)
        np.savetxt(self.dir / (self.name + "__calibrated_lite.csv"), data_lite, delimiter = ", ", header=labels_lite)

    def determine_variance_pt2pt(self, slice_size: int):
        if self.bare is not None:
            real = self.caps - self.bare
        else:
            real = self.caps
        imaginary = self.loss * self.caps
        slices = np.arange(0, len(self.time), slice_size)
        slices[-1] = len(self.time)

        std_devs_re = np.empty_like(self.time)
        std_devs_im = np.empty_like(self.time)
        for ff in range(self.fnum):
            time = self.time[:, ff]
            capacitance_shift = real[:, ff]
            imaginary_capacitance = imaginary[:, ff]
            for ss in range(len(slices) - 1):
                time_slice = time[slices[ss] : slices[ss + 1]]
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
            real = self.caps - self.bare
        else:
            real = self.caps
        imaginary = self.loss * self.caps
        slices = np.arange(0, len(self.time), slice_size)
        if len(self.time) - slices[-1] > poly_order:
            slices = np.append(slices, len(self.time))
        else:
            slices[-1] = len(self.time)
        
        std_devs_re = np.empty_like(self.time)
        std_devs_im = np.empty_like(self.time)
        for ff in range(self.fnum):
            time = self.time[:, ff]
            capacitance_shift = real[:, ff]
            imaginary_capacitance = imaginary[:, ff]
            for ss in range(len(slices) - 1):
                time_slice = time[slices[ss] : slices[ss + 1]]
                capt_slice = capacitance_shift[slices[ss] : slices[ss + 1]]
                loss_slice = imaginary_capacitance[slices[ss] : slices[ss + 1]]
                params_re[0] = np.average(capt_slice)
                params_im[0] = np.average(loss_slice)
                fit_capt = least_squares(RawData.residuals, params_re, args=(time_slice, capt_slice), method="lm")
                fit_loss = least_squares(RawData.residuals, params_im, args=(time_slice, loss_slice), method="lm")
                curve_slice_c = RawData.poly_fit(fit_capt.x, time_slice)
                curve_slice_l = RawData.poly_fit(fit_loss.x, time_slice)
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


def fit_cap(file: Path, start_clip: int = 0, end_clip: int=0):
    dataset = RawData(file)
    if end_clip:
        time = dataset.get_times()[start_clip:-end_clip, :]
        temp = dataset.get_temperatures()[start_clip:-end_clip, :]
        caps = dataset.get_capacitances()[start_clip:-end_clip, :]
        loss = dataset.get_losses()[start_clip:-end_clip, :]
    else:
        time = dataset.get_times()[start_clip:, :]
        temp = dataset.get_temperatures()[start_clip:, :]
        caps = dataset.get_capacitances()[start_clip:, :]
        loss = dataset.get_losses()[start_clip:, :]
    initial_params_cap = [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2,
                          1e-3, 1e-4, 1e-5, 1e-6, 1e-7, #1e-8, 1e-12
                          ]
    fit_result_cap = least_squares(residuals_cap, initial_params_cap, args=(temp, caps), method="lm")
    initial_params_los = [1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5,
                          1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 
                          5, 5, 5, 5, 5, 5, 5, 
                          #15, 15, 15, 15, 15, 15, 15, 
                          1e-8, 1e-10, # 1e-12
                          ]
    fit_result_los = least_squares(residuals_los, initial_params_los, args=(temp, loss), method="lm")
    print("Capacitance Fit")
    print(fit_result_cap)
    print(fit_result_cap.x)
    print("\nLoss Fit")
    print(fit_result_los)
    print(fit_result_los.x)

    print("Capacitances at room temperature:")
    for ii, f in enumerate(dataset.freqs):
        print(f" - C{subscript[0]} = {fit_result_cap.x[ii]:.6f} pF @ {int(f)} Hz")
    for ii in range(len(fit_result_cap.x) - dataset.freq_num):
        print(f" - C{subscript[ii + 1]} = {fit_result_cap.x[dataset.freq_num + ii]:.6e} pF/K{superscript[ii + 1]}")
    
    print("Losses at room temperature:")
    for ii, f in enumerate(dataset.freqs):
        print(f" - tan\u03B4{subscript[0]} = {fit_result_los.x[ii]:.6e} @ {int(f)} Hz")
    for ii in range(len(fit_result_los.x) - 3 * dataset.freq_num):
        print(f" - tan\u03B4{subscript[ii + 1]} = {fit_result_los.x[ii + dataset.freq_num * 3]:.6e} K\u207B{superscript[ii + 1]}")
    for ii, f in enumerate(dataset.freqs):
        print(f" - Amp = {fit_result_los.x[ii + dataset.freq_num]:.6e} @ {int(f)} Hz")
    for ii, f in enumerate(dataset.freqs):
        print(f" - \u03c3 = {fit_result_los.x[ii + dataset.freq_num * 2]:.6f} K @ {int(f)} Hz")
    
    # for ii, f in enumerate(dataset.freqs):
    #     print(f" - tan\u03B4{subscript[0]} = {fit_function_los.x[ii]:.6e}  @ {int(f)} Hz")

    x = np.linspace(4, 400, 10000)
    x = np.dstack([x] * temp.shape[1])[0]
    fit_to_plot = fit_function_cap(fit_result_cap.x, x)
    fig, ax = plt.subplots(1, 1)
    for ii in range(dataset.freq_num):
        ax.plot(temp[:, ii], caps[:, ii], "x")
        ax.plot(x, fit_to_plot[:, ii])
    ax.grid(linestyle='dotted')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Capacitance (pF)')
    fig.tight_layout()
    
    fig, ax = plt.subplots(1, 1)
    fit_to_plot = fit_function_los(fit_result_los.x, x)
    for ii in range(dataset.freq_num):
        ax.plot(temp[:, ii], loss[:, ii], "x")
        ax.plot(x, fit_to_plot[:, ii])
    ax.grid(linestyle='dotted')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Loss Tangent')
    fig.tight_layout()


def fit_function_cap(params, temperature):
    c0 = np.array([params[:temperature.shape[1]]])
    ci = params[temperature.shape[1] - 1:]
    rt = 300.
    temp_rt = temperature - rt
    poly = np.array(c0, dtype=np.float64) + ci[1] * temp_rt
    for ii in range(2, len(ci)):
        poly += ci[ii] * temp_rt ** ii
    return poly

def residuals_cap(params, temperature, capacitance):
    return (fit_function_cap(params, temperature) - capacitance).flatten("F")

def fit_function_los(params, temperature):
    # center = params[2]
    l0 = np.array([params[:temperature.shape[1]]])

    li = params[temperature.shape[1] * 3 - 1:]

    height = np.array(params[temperature.shape[1]:2*temperature.shape[1]])
    width = np.array(params[2*temperature.shape[1]:3*temperature.shape[1]])
    #center = params[3*temperature.shape[1]:4*temperature.shape[1]]
    t0 = 120.
    temp_t0 = temperature - t0
    poly = np.array(l0, dtype=np.float64) + li[1] * temp_t0
    for ii in range(2, len(li)):
        poly += li[ii] * temp_t0 ** ii
    arg = (temperature - 14) / width
    return poly + height * np.exp(-0.5 * arg * arg)

def residuals_los(params, temperature, losses):
    return (fit_function_los(params, temperature) - losses).flatten("F")

def func_plot(params, temperature):
    cap_at_rt = np.array(params[:-2], dtype=np.float64)
    slope = params[-2]# 5]
    quadratic = params[-1]# 4]
    # peak_loc = params[-3]
    # sigma_inv = 1. / params[-2]
    rt = 300.
    temp_rt = temperature - rt
    poly = cap_at_rt + slope * temp_rt + quadratic * temp_rt * temp_rt
    return poly # + erf(sigma_inv * (temperature - peak_loc)) 


def func_plot2(params, temperature):
    peak_locs = np.array(params[:7])
    loss_bases = np.array(params[7:-1])
    sigma_inv = 1. / params[-1]
    temp_offset = temperature - peak_locs
    return loss_bases + sigma_inv * root2pi_inv * np.exp(-0.5 * sigma_inv * sigma_inv * temp_offset * temp_offset)


def real_func(params, temperature):
    cap_rts = params[:-2]# 5]
    slope = params[-2]# 5]
    quadratic = params[-1]# 4]
    # peak_loc = params[-3]
    # sigma_inv = 1. / params[-2]
    data_per_freq = int(len(temperature) / len(cap_rts))
    cap0s = np.ones(len(temperature))
    rt = 300.
    temp_rt = temperature - rt
    for ii in range(len(cap_rts)):
        cap0s[ii*data_per_freq:(ii+1)*data_per_freq] *= cap_rts[ii]
    poly = cap0s + slope * temp_rt + quadratic * temp_rt * temp_rt
    return poly # + erf(sigma_inv * (temperature - peak_loc)) 

def imaj_func(params, temperature):
    peak_locs = params[:7]
    loss_bases = params[7:-1]
    sigma_inv = 1. / params[-1]
    data_per_freq = int(len(temperature) / len(peak_locs))
    peak_locs_full = np.ones(len(temperature))
    loss_bases_full = np.ones(len(temperature))
    for ii in range(len(peak_locs)):
        peak_locs_full[ii*data_per_freq:(ii+1)*data_per_freq] *= peak_locs[ii]
        loss_bases_full[ii*data_per_freq:(ii+1)*data_per_freq] *= loss_bases[ii]
    temp_offset = temperature - peak_locs_full
    return loss_bases_full + sigma_inv * root2pi_inv * np.exp(-0.5 * sigma_inv * sigma_inv * temp_offset * temp_offset)

def residuals1(params, temperature, capacitance):
    return real_func(params, temperature) - capacitance # + imaj_func(params, temperature) - loss

def residuals2(params, temperature, loss):
    return imaj_func(params, temperature) - loss


if __name__ == "__main__":
    import matplotlib.pylab as plt

    file = Path(r"H:\OneDrive - UCB-O365\Rogerslab3\Teddy\Thesis\chapter-4\Data\BDS\1@TPP sat - GBA 124\Cooling_1468959484_96 - Copy.csv")
    pow = Powder(file, 1.175)
    plt.show()
    