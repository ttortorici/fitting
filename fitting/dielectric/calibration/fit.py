from scipy.optimize import least_squares
from scipy.special import erf
from pathlib import Path
from load_data import DataSet
import numpy as np
import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


root2pi_inv = 1. / np.sqrt(2.0 * np.pi)
max_nfev = 1 # iterate one at a time


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
    initial_params = [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2,
                      1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-12]
    fit_result = least_squares(residuals, initial_params, args=(temp, caps), method="lm")
    print(fit_result)
    print(fit_result.x)
    
    x = np.linspace(0, 400, 1000)
    x = np.dstack([x] * temp.shape[1])[0]
    fit_to_plot = fit_function(fit_result.x, x)
    plt.figure()
    for ii in range(dataset.freq_num):
        plt.plot(temp[:, ii], caps[:, ii], "x")
        plt.plot(x, fit_to_plot[:, ii])


def fit_function(params, temperature):
    c0 = np.array([params[:temperature.shape[1]]])
    ci = params[temperature.shape[1] - 1:]
    rt = 300.
    temp_rt = temperature - rt
    poly = np.array(c0, dtype=np.float64) + ci[1] * temp_rt
    for ii in range(2, len(ci)):
        poly += ci[ii] * temp_rt ** ii
    return poly

def residuals(params, temperature, capacitance):
    return (fit_function(params, temperature) - capacitance).flatten("F")

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

    dir = Path("C:\\Users\\Teddy\\OneDrive - UCB-O365\\Rogerslab3\\Teddy\\TPP Films\\BTB-TPP\\2024 Film Growth\\Film 2\\BDS\\00 - Cals")
    file = dir / "2024-02-29__none__M09-1-TT501__CAL__T-13-39_rev-freq.csv"
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
    