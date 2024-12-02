from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pylab as plt
plt.style.use("fitting.style")

class RawFile:
    TIME_IND = 0
    TEMPA_IND = 1
    TEMPB_IND = 2
    CAP_IND = 3
    LOSS_IND = 4
    VOLT_IND = 5
    FREQ_IND = 6
    COLS_PER = 7

    def __init__(self, files: Path, ):
        if isinstance(files, Path) or isinstance(files, str):
            self.data = self.loadtxt(files)
        elif isinstance(files, list) or isinstance(files, tuple):
            self.data = self.loadtxt(files[0])
            for file in files[1:]:
                self.data = np.append(self.data, self.loadtxt(file), axis=0)
        self.shape = self.data.shape

        self.freq_num = int(self.shape[1] / self.COLS_PER)
        self.freqs = [self.data[0, index] for index in self.get_inds(self.FREQ_IND)]
        self.cap_std, self.loss_std = self.determine_variance(10, 1)

    def determine_variance(self, slice_size: int, poly_order: int):
        params_c = [0] * (poly_order + 1)
        params_l = params_c.copy()
        times = self.get_times()
        capacitances = self.get_capacitances()
        losstangents = self.get_losses()
        slices = np.arange(0, len(times), slice_size)
        if len(times) - slices[-1] > poly_order:
            slices = np.append(slices, len(times))
        else:
            slices[-1] = len(times)
        
        std_devs_c = np.empty_like(times)
        std_devs_l = np.empty_like(times)
        for ff, freq in enumerate(self.freqs):
            time = times[:, ff]
            capacitance = capacitances[:, ff]
            losstangent = losstangents[:, ff]
            for ss in range(len(slices) - 1):
                time_slice = time[slices[ss] : slices[ss + 1]]
                capt_slice = capacitance[slices[ss] : slices[ss + 1]]
                loss_slice = losstangent[slices[ss] : slices[ss + 1]]
                params_c[0] = np.average(capt_slice)
                params_l[0] = np.average(loss_slice)
                fit_capt = least_squares(self.residuals, params_c, args=(time_slice, capt_slice), method="lm")
                fit_loss = least_squares(self.residuals, params_l, args=(time_slice, loss_slice), method="lm")
                curve_slice_c = self.poly_fit(fit_capt.x, time_slice)
                curve_slice_l = self.poly_fit(fit_loss.x, time_slice)
                diff_c = curve_slice_c - capt_slice
                diff_l = curve_slice_l - loss_slice

                variance_c = np.sum(diff_c * diff_c) / (len(capt_slice) - 1)
                variance_l = np.sum(diff_l * diff_l) / (len(loss_slice) - 1)
                std_devs_c[slices[ss]: slices[ss + 1], ff] = np.sqrt(variance_c)
                std_devs_l[slices[ss]: slices[ss + 1], ff] = np.sqrt(variance_l)
        return std_devs_c, std_devs_l
    
    @staticmethod
    def poly_fit(params: list, temperature: np.ndarray) -> np.ndarray:
        center = 0.5 * (temperature.max() - temperature.min())
        temp_rt = temperature - center
        poly = params[0]
        for ii, p in enumerate(params[1:]):
            poly += p * temp_rt ** (ii + 1)
        return poly
    
    @classmethod
    def residuals(cls, params: list, x_data: np.ndarray, y_data: np.ndarray):
        return (cls.poly_fit(params, x_data) - y_data).flatten("F")
    
    def get_inds(self, col_ind):
        return [col_ind + self.COLS_PER * ii for ii in range(self.freq_num)]
    
    def get_times(self):
        inds = self.get_inds(self.TIME_IND)
        return self.data[:, inds]
    
    def get_temperatures(self):
        inds = self.get_inds(self.TEMPA_IND)
        return self.data[:, inds]
    
    def get_shield_temperatures(self):
        inds = self.get_inds(self.TEMPB_IND)
        return self.data[:, inds]
    
    def get_capacitances(self):
        inds = self.get_inds(self.CAP_IND)
        return self.data[:, inds]
    
    def get_losses(self):
        inds = self.get_inds(self.LOSS_IND)
        return self.data[:, inds]
    
    def get_voltages(self):
        inds = self.get_inds(self.VOLT_IND)
        return self.data[:, inds]
    
    def plot(self, figsize=None, vertical=True, real_imaginary=False):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        temperature = self.get_temperatures()
        capacitance = self.get_capacitances()
        losstangent = self.get_losses()
        print(capacitance.shape)
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
                temperature[:, ff],
                capacitance[:, ff],
                s=5,
                facecolor="w",
                edgecolor=colors[ff % len(colors)],
                lw=.75,
                label=freq_str,
            )
        ax_im.set_xlabel("Temperature (K)")
        if real_imaginary:
            imaginary_cap = capacitance * losstangent
            ax_re.set_ylabel("$C'$ (pF)")
            ax_im.set_ylabel("$C''$ (pF)")
            for ff, freq in enumerate(self.freqs):
                freq_str = str(int(freq))
                if len(freq_str) > 4:
                    freq_str = f"{freq_str[:-3]} kHz"
                else:
                    freq_str += " Hz"
                ax_im.scatter(
                    temperature[:, ff],
                    imaginary_cap[:, ff],
                    s=5,
                    facecolor="w",
                    edgecolor=colors[ff % len(colors)],
                    lw=.75,
                    label=freq_str,
                )
        else:
            ax_re.set_ylabel("Capacitance (pF)")
            ax_im.set_ylabel("$\\tan\\delta$")
            for ff, freq in enumerate(self.freqs):
                freq_str = str(int(freq))
                if len(freq_str) > 4:
                    freq_str = f"{freq_str[:-3]} kHz"
                else:
                    freq_str += " Hz"
                ax_im.scatter(
                    temperature[:, ff],
                    losstangent[:, ff],
                    s=5,
                    facecolor="w",
                    edgecolor=colors[ff % len(colors)],
                    lw=.75,
                    label=freq_str,
                )
        ax_re.grid()
        ax_im.grid()
        ax_im.legend()
        fig.tight_layout()
        


    @staticmethod
    def loadtxt(filename: str) -> np.ndarray:
        return np.loadtxt(filename, comments="#", delimiter=",", skiprows=3, dtype=np.float64)


class ProcessedFile(RawFile):
    LABELS = ["Time [s]", "Temperature A [K]", "Temperature B [K]",
              "Capacitance [pF]", "Cap STD [pF]",
              "Bare Cap Curves [pF]", "Bare Cap STD [pF]",
              "Delta C' [pF]", "Delta C' STD [pF]",
              "Loss Tangent", "Loss Tangent STD",
              "Bare Loss", "Bare Loss STD",
              "C'' [pF]", "C'' STD [PF]",
              "Bare C'' [pF]", "Bare C'' STD [pF]",
              "Delta C'' [pF]", "Delta C'' STD [pF]",
              "Real Susceptibility", "Real Susceptibility STD",
              "Imaginary Susceptibility", "Imaginary Susceptibility STD",
              "Voltage [V]", "Frequency [Hz]"]

    TIME_IND = 0
    TEMPA_IND = 1
    TEMPB_IND = 2
    CAP_IND = 3         # measured capacitance
    CAPERR_IND = 4
    BAREC_IND = 5       # fitted bare capacitance
    BARECERR_IND = 6
    DELCRE_IND = 7      # delta C'
    DELCREERR_IND = 8
    LOSS_IND = 9        # measured loss
    LOSSERR_IND = 10
    BAREL_IND = 11      # fitted bare loss
    BARELERR_IND = 12
    CAPIM_IND = 13      # C''
    CAPIMERR_IND = 14
    BARECIM_IND = 15    # bare C''
    BARECIMERR_IND = 16
    DELCIM_IND = 17     # delta C''
    DELCIMERR_IND = 18
    CHIRE_IND = 19      # electric susceptibility (real part)
    CHIREERR_IND = 20
    CHIIM_IND = 21      # electric susceptibility (imaginary part)
    CHIIMERR_IND = 22
    VOLT_IND = 23
    FREQ_IND = 24
    COLS_PER = 25

    def get_capacitance_errors(self):
        inds = self.get_inds(self.CAPERR_IND)
        return self.data[:, inds]

    def get_bare_capacitances(self):
        inds = self.get_inds(self.BAREC_IND)
        return self.data[:, inds]
    
    def get_bare_capacitances_errors(self):
        inds = self.get_inds(self.BARECERR_IND)
        return self.data[:, inds]
    
    def get_capacitance_shifts_real(self):
        inds = self.get_inds(self.DELCRE_IND)
        return self.data[:, inds]
    
    def get_capacitance_shift_errors_real(self):
        inds = self.get_inds(self.DELCREERR_IND)
        return self.data[:, inds]
    
    def get_loss_errors(self):
        inds = self.get_inds(self.LOSSERR_IND)
        return self.data[:, inds]
    
    def get_bare_losses(self):
        inds = self.get_inds(self.BAREL_IND)
        return self.data[:, inds]
    
    def get_bare_loss_errors(self):
        inds = self.get_inds(self.BARELERR_IND)
        return self.data[:, inds]
    
    def get_imaginary_capacitances(self):
        inds = self.get_inds(self.CAPIM_IND)
        return self.data[:, inds]
    
    def get_imaginary_capacitance_errors(self):
        inds = self.get_inds(self.CAPIMERR_IND)
        return self.data[:, inds]
    
    def get_imaginary_bare_capacitances(self):
        inds = self.get_inds(self.BARECIM_IND)
        return self.data[:, inds]
    
    def get_imaginary_bare_capacitance_errors(self):
        inds = self.get_inds(self.BARECIMERR_IND)
        return self.data[:, inds]
    
    def get_imaginary_capacitance_shifts(self):
        inds = self.get_inds(self.DELCIM_IND)
        return self.data[:, inds]
    
    def get_imaginary_capacitance_shift_errors(self):
        inds = self.get_inds(self.DELCIMERR_IND)
        return self.data[:, inds]
    
    def get_real_susceptibilities(self):
        inds = self.get_inds(self.CHIRE_IND)
        return self.data[:, inds]
    
    def get_real_susceptibility_errors(self):
        inds = self.get_inds(self.CHIRE_IND)
        return self.data[:, inds]
    
    def get_imaginary_susceptibilities(self):
        inds = self.get_inds(self.CHIRE_IND)
        return self.data[:, inds]
    
    def get_imaginary_susceptibility_errors(self):
        inds = self.get_inds(self.CHIRE_IND)
        return self.data[:, inds]
    

class ProcessedFileLite(ProcessedFile):
    LABELS = ["Time [s]", "Temperature A [K]",
              "Delta C' [pF]", "Delta C' STD [pF]",
              "Delta C'' [pF]", "Delta C'' STD [pF]",
              "Real Susceptibility", "Real Susceptibility STD",
              "Imaginary Susceptibility", "Imaginary Susceptibility STD",
              "Voltage [V]", "Frequency [Hz]"]

    TIME_IND = 0
    TEMPA_IND = 1
    DELCRE_IND = 2      # delta C'
    DELCREERR_IND = 3
    DELCIM_IND = 4      # delta C''
    DELCIMERR_IND = 5
    CHIRE_IND = 6       # electric susceptibility (real part)
    CHIREERR_IND = 7
    CHIIM_IND = 8       # electric susceptibility (imaginary part)
    CHIIMERR_IND = 9
    VOLT_IND = 10
    FREQ_IND = 11
    COLS_PER = 12
    TEMPB_IND = None
    CAP_IND = None
    CAPERR_IND = None
    BAREC_IND = None
    BARECERR_IND = None
    LOSS_IND = None
    LOSSERR_IND = None
    BAREL_IND = None
    BARELERR_IND = None
    CAPIM_IND = None
    CAPIMERR_IND = None
    BARECIM_IND = None
    BARECIMERR_IND = None
    

class RawData(RawFile):

    def __init__(self, files):
        super().__init__(files)
        # self.determine_ascending()
        self.time_derivative_filter()

    def determine_ascending(self):
        derivative = self.calc_time_derivatives()
        ascending = np.all(derivative >= 0, axis=1)
        return ascending
    
    def calc_time_derivatives(self):
        time_inds = self.get_inds(RawData.TIME_IND)
        temp_inds = self.get_inds(RawData.TEMPA_IND)
        derivative = (self.data[1:, temp_inds] - self.data[:-1, temp_inds]) / (self.data[1:, time_inds] - self.data[:-1, time_inds])
        return np.vstack((derivative, np.zeros(self.freq_num)))
    
    def time_derivative_filter(self):
        print(f"filtering data with shape: {self.data.shape}")

        derivative = self.calc_time_derivatives()
        is_close = np.logical_not(np.all(np.isclose(derivative, 0, atol=1e-3, rtol=2e-3), axis=1))
        self.data = self.data[is_close]
        self.cap_std = self.cap_std[is_close]
        self.loss_std = self.loss_std[is_close]
        print(f"now is shape: {self.data.shape}")
        self.shape = self.data.shape


def reverse_freqs_in_data_set(file: Path):
    data_set = RawData(file)
    data = data_set.data
    new_data = np.empty(data.shape)
    for ii in range(data_set.freq_num):
        start_placement = ii * RawData.COLS_PER
        end_placenemt = start_placement + RawData.COLS_PER
        end_grab = (data_set.freq_num - ii) * RawData.COLS_PER
        start_grab = end_grab - RawData.COLS_PER
        new_data[:, start_placement:end_placenemt] = data[:, start_grab:end_grab]
    
    header_lines = ""
    with open(file, 'r') as f:
        for ii in range(3):
            header_lines += f.readline().lstrip("# ")
    header_lines = header_lines.rstrip("\n")
    new_filename = file.parent / (file.name.rstrip(".csv") + "_rev-freq.csv")
    np.savetxt(str(new_filename), new_data, delimiter=",", header=header_lines)


if __name__ == "__main__":
    import matplotlib.pylab as plt
    params = {
        "mathtext.fontset": "cm",
        "font.family": "STIXGeneral",
        "font.size": 12,
        "xtick.top": True,
        "ytick.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "grid.linestyle": "dotted",
    }
    plt.rcParams.update(params)

    path = Path(r"C:\Users\Teddy\OneDrive - UCB-O365\Rogerslab3\Teddy\TPP Films\BAP-TPP\Film 20241014\Dielectric")
    bare_file = "2024-10-11__bare__M01-2__CAL__T-12-16.csv"
    film_file = "2024-10-15__BAPaTPP2-27C__M01-2__FILM__T-08-58.csv"
    data = RawData(path / film_file)

    # dir = Path("C:\\Users\\Teddy\\OneDrive - UCB-O365\\Rogerslab3\\Teddy\\TPP Films\\BTB-TPP\\2024 Film Growth\\Film 2\\BDS\\00 - Cals")
    # file = dir / "2024-02-29__none__M09-1-TT501__CAL__T-13-39_rev-freq.csv"
    # data = DataSet(file)
    # # print(data.data)
    # print(data.shape)
    # print(data.freqs)
    # time = data.get_times()
    # temp = data.get_temperatures()
    # caps = data.get_capacitances()
    # freqs = data.freqs
    # loss = data.get_losses()
    #start = 100
    # plt.figure()
    # for ii in range(data.freq_num):
    #     plt.plot(temp[start:, ii], caps[start:, ii])
    # plt.figure()
    # for ii in range(data.freq_num):
    #     plt.plot(temp[start:, ii], loss[start:, ii])
    # plt.figure()
    # for ii in range(data.freq_num):
    #     plt.scatter(time[:, ii], temp[:, ii], s=2)
    # time = np.average(data.get_times(), axis=1)
    # plt.scatter(time, data.determine_ascending() * 100, s=5)

    data.plot(real_imaginary=True)

    plt.show()
    
