from fitting.data import DataSet
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pylab as plt
plt.style.use("fitting.style")


class Bare:
    
    def __init__(self, data: DataSet, c300: float=1., linear: float=1e-5, quadratic: float=1e-8):
        """
        :param data: DataSet object containing the data
        :param c300: capacitance at 300 K
        :param linear: linear dependance of capacitance on temperature pF per K
        :param quadratic: quadratic dependance of capacitance on temperature pF per K^2
        """
        self.freq_num = data.freq_num
        self.c300 = np.ones(self.freq_num) * c300
        self.linear = linear
        self.quadratic = quadratic
        self.temperature = data.get_temperatures()
        self.capacitance = data.get_capacitances()
        self.frequencies = data.frequencies
        self.reverse = data.reverse
        self.COLORS = data.COLORS
        self.temperature_fit = np.stack(
            [np.linspace(self.temperature.min()-10, self.temperature.max()+10, 1000)] * self.freq_num
        ).T
        self.errors = None
        
    @staticmethod
    def function(temperature:np.ndarray, c300: np.ndarray, linear: float, quadratic: float):
        """
        :param temperature: array of size (data_points, freq_num)
        :param c300: capacitance at room temperature. array of size (freq_num,)
        :linear: pF per K
        :quadratic: pF per K^2
        :return: C(T)
        """
        temp = temperature - 300
        return c300 + temp * linear + temp * temp * quadratic
    
    def result(self, temperature):
        return self.function(temperature, self.c300, self.linear, self.quadratic)
    
    def fit(self):
        params = list(self.c300)
        params.extend([self.linear, self.quadratic])

        def residuals(params):
            c300 = np.array(params[:self.freq_num])
            linear = params[-2]
            quadratic = params[-1]
            cap_hat = self.function(self.temperature, c300, linear, quadratic)
            return (cap_hat - self.capacitance).flatten()
        
        result = least_squares(residuals, params, method="lm", ftol=1e-9, xtol=1e-9, gtol=1e-9)
        print(result)
        print("")
        self.c300 = np.array(result.x[:self.freq_num])
        self.linear = result.x[-2]
        self.quadratic = result.x[-1]
        self.errors = np.sqrt(np.diagonal(np.linalg.inv(result.jac.T @ result.jac) * (result.fun.T @ result.fun / (result.fun.size - result.x.size))))
        for ii in range(self.freq_num):
            print(f"C = ({self.c300[ii]:.6f} \u00B1 {self.errors[ii]:.6f}) pF at {self.frequencies[ii]} Hz")
        print(f"linear term is ({self.linear:.3e} \u00B1 {self.errors[-2]:.3e}) pF/K")
        print(f"quadtratic term is ({self.quadratic:.3e} \u00B1 {self.errors[-1]:.3e}) pF/K^2")
    
    def show(self):
        if self.freq_num == 3:
            colors = self.COLORS[::3]
        else:
            colors = self.COLORS
        if self.reverse:
            colors = colors[::-1]

        fit = self.function(self.temperature_fit, self.c300, self.linear, self.quadratic)

        fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
        ax.grid()
        for ii in range(self.freq_num):
            if self.reverse:
                ii = self.freq_num - ii - 1
            freq_name = str(int(self.frequencies[ii]))
            if len(freq_name) > 3:
                freq_name = freq_name[:-3] + " kHz"
            else:
                freq_name += " Hz"
            ax.scatter(
                self.temperature[:, ii],
                self.capacitance[:, ii],
                s=4,
                marker="o",
                edgecolors=colors[ii],
                lw=.75,
                alpha=1,
                facecolor='w',
                label=freq_name
            )
            ax.plot(self.temperature_fit[:, ii], fit[:, ii], 'r', linewidth=1)
        ax.set_title("Bare Capacitor")
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Capacitance (pF)")
        fig.tight_layout()
        return fig, ax
