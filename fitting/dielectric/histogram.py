import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit, least_squares
import matplotlib.pylab as plt
plt.style.use('fitting.style')


class LoadFit:

    def __init__(self, file: Path):
        self.center = np.empty(3, dtype=np.float64)
        self.center_err = np.empty(3, dtype=np.float64)
        self.spacing = np.empty(3, dtype=np.float64)
        self.spacing_err = np.empty(3, dtype=np.float64)
        self.amplitudes = np.empty((11, 3), dtype=np.float64)
        self.amplitudes_err = np.empty((11, 3), dtype=np.float64)
        self.asymmetry = np.empty(3, dtype=np.float64)
        self.asymmetry_err = np.empty(3, dtype=np.float64)
        self.attempt = np.empty(3, dtype=np.float64)
        self.attempt_err = np.empty(3, dtype=np.float64)
        self.cb300 = None
        start = False
        with open(file, "r") as f:
            for line in f.readlines():
                if start:
                    if 'freq' in line:
                        break
                    self.parse_line(line)
                    
                if '\t\tValue\t' in line:
                    start = True

    def parse_line(self, line):
        list_line = line.split('\t')
        param = list_line[1].strip("*")
        if param[0] in ["e", "g", "a", "s", "t", "c"]:
            value = float(list_line[2])
            error = float(list_line[3])
        else:
            return
        if len(param) == 2 and param[1] in ["1", "2", "3", "0"]:
            peak_index = int(param[1]) - 1
            if param[0] == "e":
                self.center[peak_index] = value
                self.center_err[peak_index] = error
            elif param[0] == "g":
                self.spacing[peak_index] = value
                self.spacing_err[peak_index] = error
            elif param[0] == "s":
                self.asymmetry[peak_index] = value
                self.asymmetry_err[peak_index] = error
            elif param[0] == "t":
                if peak_index:
                    self.attempt[peak_index] = value
                    self.attempt_err[peak_index] = error
                else:
                    self.attempt = value
                    self.attempt_err = error
        elif "amp" in param:
            peak_index = int(param[3]) - 1
            if param[4] == "c":
                self.amplitudes[5, peak_index] = value
                self.amplitudes_err[5, peak_index] = error
            else:
                dist_num = int(param[4:]) - 1
                if dist_num >= 5:
                    dist_num += 1
                self.amplitudes[dist_num, peak_index] = value
                self.amplitudes_err[dist_num, peak_index] = error
        elif "ce300" in param:
            self.cb300 = value
        self.amplitudes_err[np.where(self.amplitudes_err == 0)] = 1

    @staticmethod
    def gaussian(x, amp, sigma, x0):
        arg = (x - x0) / sigma
        return amp * np.exp(-0.5 * arg * arg)
    
    @classmethod
    def gaussian2(cls, x, y, spacing, sigma, x0):
        area = np.sum(y) * spacing
        amp = area / (np.sqrt(2. * np.pi) * sigma)
        return cls.gaussian(x, amp, sigma, x0)
    
    @classmethod
    def residuals(cls, params, x, y, spacing, weight):
        return (y - cls.gaussian2(x, y, spacing, *params)) / weight


    def hist(self, plot_curves=True):
        for ii in range(len(self.center)):
            print(f"PEAK {ii+1}")
            print(f" - total area : {np.sum(self.amplitudes[:, ii] * self.spacing[ii])}")
            energies = self.center[ii] + self.spacing[ii] * np.arange(-5, 6, 1)
            try:
                popt1, pcov1 = curve_fit(self.gaussian, energies, self.amplitudes[:, ii], p0=(1, self.spacing[ii], self.center[ii]))
            except RuntimeError:
                popt1 = None
            try:
                fit2 = least_squares(self.residuals, x0=(self.spacing[ii], self.center[ii]),
                                     args=(energies, self.amplitudes[:, ii], self.spacing[ii], self.amplitudes_err[:, ii]))
            except RuntimeError:
                fit2 = None
            x = np.linspace(energies.min(), energies.max(), 100)
            fig, ax = plt.subplots(1, 1, figsize=(5,4))
            ax.bar(energies, self.amplitudes[:, ii], width=self.spacing[ii]*.9)
            if popt1 is not None:
                print(f" - curve1 area : {np.sqrt(2 * np.pi) * popt1[1] * popt1[0]}")
                print(f" - curve1 center: {popt1[-1]}")
                print(f" - curve1 sigma : {popt1[1]}")
                if plot_curves:
                    ax.plot(x, self.gaussian(x, *popt1), "orange")
            if fit2 is not None:
                print(f" - curve2 center: {fit2.x[1]}")
                print(f" - curve2 sigma : {fit2.x[0]}")
                if plot_curves:
                    ax.plot(x, self.gaussian2(x, self.amplitudes[:, ii], self.spacing[ii], *fit2.x), "r")
            ax.set_xlabel("activation energy")
            ax.set_ylabel("amplitudes")
            ax.grid()
            fig.tight_layout()
            # fig.savefig(f"peak-{ii+1}-amplitudes.png", dpi=100, bbox_inches="tight")


if __name__ == "__main__":
    file = Path(r"C:\Users\Teddy\OneDrive - UCB-O365\Rogerslab3\Teddy\Thesis\chapter-4\Data\BDS\1@TPP sat - GBA 124\origin-fit_results.txt")
    fit = LoadFit(file)
    fit.hist()
    plt.show()