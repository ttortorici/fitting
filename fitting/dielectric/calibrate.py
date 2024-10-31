from scipy.optimize import least_squares
from scipy.special import erf
from pathlib import Path
from fitting.dielectric.load import DataSet
import numpy as np
import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


root2pi_inv = 1. / np.sqrt(2.0 * np.pi)
max_nfev = 1 # iterate one at a time

subscript = ["\u2080", "\u2081", "\u2082", "\u2083", "\u2084",
             "\u2085", "\u2086", "\u2087", "\u2088", "\u2089"]
superscript = ["\u2070", "\u00B9", "\u00B2", "\u00B3", "\u2074",
               "\u2075", "\u2076", "\u2077", "\u2078", "\u2079"]


class Bare:
    def __init__(self):
        self.time = None
        self.temp = None
        self.caps = None
        self.loss = None
        self.freq = None
        self.fnum = None
        self._fit_real = None
        self._fit_imag = None
    
    def fit_real(self, poly_order: int, room_temperature_measurement: float=1.2):
        if self.time is None:
            raise AttributeError("Need to load data first with .load_bare(file)")
        initial_params = [room_temperature_measurement] * self.fnum
        initial_params.extend(list(np.logspace(-3, -3-poly_order, poly_order - 1)))

        fit_result = least_squares(self.residuals_real, initial_params, args=(self.temp, self.caps), method="lm")
        print("Capacitance Fit")
        print(fit_result)
        self._fit_real = fit_result.x
        
    def fit_imag(self, poly_order: int, room_temperature_measurement: float = 1e-5):
        if self.time is None:
            raise AttributeError("Need to load data first with .load_bare(file)")
        initial_params = [room_temperature_measurement] * (2 * self.fnum)
        initial_params.extend([5] * self.fnum)
        initial_params.extend(list(np.logspace(-8, -8-poly_order, poly_order - 1)))

        fit_result = least_squares(self.residuals_imag, initial_params, args=(self.temp, self.loss), method="lm")
        print("Loss Fit")
        print(fit_result)
        self._fit_imag = fit_result.x

    @staticmethod
    def fit_function_real(params, temperature):
        c0 = np.array([params[:temperature.shape[1]]])
        ci = params[temperature.shape[1] - 1:]
        rt = 300.
        temp_rt = temperature - rt
        poly = np.array(c0, dtype=np.float64) + ci[1] * temp_rt
        for ii in range(2, len(ci)):
            poly += ci[ii] * temp_rt ** ii
        return poly

    @classmethod
    def residuals_real(cls, params, temperature, capacitance):
        return (cls.fit_function_real(params, temperature) - capacitance).flatten("F")

    @staticmethod
    def fit_function_imag(params, temperature):
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

    @classmethod
    def residuals_imag(cls, params, temperature, losses):
        return (cls.fit_function_imag(params, temperature) - losses).flatten("F")
    
    def report_fits(self):
        if self._fit_real is None:
            print("Capacitances have not been fitted.")
        else:
            print("Capacitances at room temperature:")
            for ii, f in enumerate(self.freq):
                print(f" - C{subscript[0]} = {self._fit_real[ii]:.6f} pF @ {int(f)} Hz")
            for ii in range(len(self._fit_real) - self.fnum):
                print(f" - C{subscript[ii + 1]} = {self._fit_real[self.fnum + ii]:.6e} pF/K{superscript[ii + 1]}")
        if self._fit_imag is None:
            print("Losses have not been fitted.")
        else:
            print("Losses at room temperature:")
            for ii, f in enumerate(self.freq):
                print(f" - tan\u03B4{subscript[0]} = {self._fit_imag[ii]:.6e} @ {int(f)} Hz")
            for ii in range(len(self._fit_imag) - 3 * self.fnum):
                print(f" - tan\u03B4{subscript[ii + 1]} = {self._fit_imag[ii + self.fnum * 3]:.6e} K\u207B{superscript[ii + 1]}")
            for ii, f in enumerate(self.freq):
                print(f" - Amp = {self._fit_imag[ii + self.fnum]:.6e} @ {int(f)} Hz")
            for ii, f in enumerate(self.freq):
                print(f" - \u03c3 = {self._fit_imag[ii + self.fnum * 2]:.6f} K @ {int(f)} Hz")

    def load_bare(self, file: Path, start_clip: int = 0, end_clip: int=0):
        dataset = DataSet(file)
        if end_clip:
            self.time = dataset.get_times()[start_clip:-end_clip, :]
            self.temp = dataset.get_temperatures()[start_clip:-end_clip, :]
            self.caps = dataset.get_capacitances()[start_clip:-end_clip, :]
            self.loss = dataset.get_losses()[start_clip:-end_clip, :]
        else:
            self.time = dataset.get_times()[start_clip:, :]
            self.temp = dataset.get_temperatures()[start_clip:, :]
            self.caps = dataset.get_capacitances()[start_clip:, :]
            self.loss = dataset.get_losses()[start_clip:, :]
        self.freq = dataset.freqs
        self.fnum = dataset.freq_num

    def show_fit(self):
        x = np.linspace(4, 400, 10000)
        x = np.dstack([x] * self.temp.shape[1])[0]
        fit_to_plot = fit_function_cap(fit_result_cap.x, x)
        fig, ax = plt.subplots(1, 1)
        for ii in range(dataset.freq_num):
            ax.plot(temp[:, ii], caps[:, ii], "x")
            ax.plot(x, fit_to_plot[:, ii])
        ax.grid(linestyle='dotted')
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Capacitance (pF)')
        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        fig.tight_layout()

        fig, ax = plt.subplots(1, 1)
        fit_to_plot = fit_function_los(fit_result_los.x, x)
        for ii in range(dataset.freq_num):
            ax.plot(temp[:, ii], loss[:, ii], "x")
            ax.plot(x, fit_to_plot[:, ii])
        ax.grid(linestyle='dotted')
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Loss Tangent')
        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        fig.tight_layout()


def fit_cap(file: Path, start_clip: int = 0, end_clip: int=0):
    dataset = DataSet(file)
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
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    fig.tight_layout()
    
    fig, ax = plt.subplots(1, 1)
    fit_to_plot = fit_function_los(fit_result_los.x, x)
    for ii in range(dataset.freq_num):
        ax.plot(temp[:, ii], loss[:, ii], "x")
        ax.plot(x, fit_to_plot[:, ii])
    ax.grid(linestyle='dotted')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Loss Tangent')
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
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

    # dir = Path("C:\\Users\\Teddy\\OneDrive - UCB-O365\\Rogerslab3\\Teddy\\TPP Films\\BTB-TPP\\2024 Film Growth\\Film 2\\BDS\\00 - Cals")
    # file = dir / "2024-02-29__none__M09-1-TT501__CAL__T-13-39_rev-freq.csv"
    file = Path(r"C:\Users\Teddy\OneDrive - UCB-O365\Rogerslab3\Teddy\TPP Films\BAP-TPP\Film 20241014\Dielectric\2024-10-11__bare__M01-2__CAL__T-12-16.csv")
    fit_cap(file)  #, 300, 200)
    plt.show()
    # print(data.data)
    # print(data.shape)
    # print(data.freqs)
    # time = data.get_times()
    # temp = data.get_temperatures()
    # caps = data.get_capacitances()
    # loss = data.get_losses()
    # start = 100


    # plt.figure()
    # for ii in range(data.freq_num):
    #     plt.plot(temp[start:, ii], caps[start:, ii])
    # plt.figure()
    # for ii in range(data.freq_num):
    #     plt.plot(temp[start:, ii], loss[start:, ii])
    # plt.figure()
    # for ii in range(data.freq_num):
    #     plt.plot(time[start:, ii], temp[start:, ii])
    # 
    # plt.show()
    