import numpy as np
from scipy.optimize import least_squares


def linear_regression(x, y):
    pts = len(x)
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)
    cross_deviation = np.sum(y * x, axis=0) - pts * x_mean * y_mean
    deviation_about_x = np.sum(x * x, axis=0) - pts * x_mean * x_mean
    linear = cross_deviation / deviation_about_x
    offset = y_mean - linear * x_mean
    return offset, linear


def calculate(x, real, imag, slice_size=8):
    slices = np.arange(0, len(x), slice_size)
    if len(x) - slices[-1] > 1:
        slices = np.append(slices, len(x))
    else:
        slices[-1] = len(x)
    std_devs_real = np.empty_like(x)
    std_devs_imag = np.empty_like(x)
    for ss in range(len(slices) - 1):
        x_slice = x[slices[ss] : slices[ss + 1], :]
        re_slice = real[slices[ss] : slices[ss + 1], :]
        im_slice = imag[slices[ss] : slices[ss + 1], :]
        offset_re, linear_re = linear_regression(x_slice, re_slice)
        offset_im, linear_im = linear_regression(x_slice, im_slice)
        residuals_re = offset_re + x_slice * linear_re - re_slice
        residuals_im = offset_im + x_slice * linear_im - im_slice
        variance_re = np.sum(residuals_re * residuals_re, axis=0) / (len(re_slice) - 1)
        variance_im = np.sum(residuals_im * residuals_im, axis=0) / (len(im_slice) - 1)
        std_devs_real[slices[ss]: slices[ss + 1], :] = np.sqrt(variance_re)
        std_devs_imag[slices[ss]: slices[ss + 1], :] = np.sqrt(variance_im)
    return std_devs_real, std_devs_imag

if __name__ == "__main__":
    import matplotlib.pylab as plt
    plt.style.use("fitting.style")

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    x = np.linspace(0, 1, 100)
    x = np.column_stack((x, x, x))
    y = 2 * np.sin(x) + np.random.random(x.shape) + np.array([0, 2, 4]) + x * x * np.random.random(x.shape) * 5

    offsets, linears = linear_regression(x, y)
    y_hat = x * linears + offsets
    standard_dev, _ = calculate(x, y, y)

    fig, ax = plt.subplots(1, 1)
    
    for ii in range(x.shape[1]):
        ax.errorbar(x[:, ii], y[:, ii], yerr=standard_dev[:, ii], fmt='o',
                    ecolor=colors[ii], markerfacecolor='w', markeredgewidth=1.5, capsize=3)
        ax.plot(x[:, ii], y_hat[:, ii], colors[ii])
    
    plt.show()