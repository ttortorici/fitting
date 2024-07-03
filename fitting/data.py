from pathlib import Path
import numpy as np
import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams["mathtext.fontset"] = "cm"


class DataSet:
    TIME_IND = 0
    TMPA_IND = 1
    TMPB_IND = 2
    CAPT_IND = 3
    LOSS_IND = 4
    VOLT_IND = 5
    FREQ_IND = 6
    COLS_PER = 7

    COLORS = ["k", "darkgreen", "turquoise", "b", "slateblue", "darkviolet", "r"]

    def __init__(self, files, name=""):
        if isinstance(files, Path) or isinstance(files, str):
            self.data = self.loadtxt(files)
        elif isinstance(files, list) or isinstance(files, tuple):
            self.data = self.loadtxt(files[0])
            for file in files[1:]:
                self.data = np.concatenate((self.data, self.loadtxt(file)), axis=0)
        self.shape = self.data.shape
        self.name = name
        print(f"Loaded data with shape: {self.shape}.")

        self.freq_num = int(self.shape[1] / self.COLS_PER)
        self.frequencies = [self.data[0, index] for index in self.get_inds(self.FREQ_IND)]

        print(f"{tuple(self.frequencies)} frequencies found")
        
        if self.frequencies[0] > self.frequencies[-1]:
            self.reverse = True
        else:
            self.reverse = False
        self.time_derivative_filter()
    
    def get_frequencies(self):
        return self.frequencies

    def get_inds(self, col_ind):
        return [col_ind + self.COLS_PER * ii for ii in range(self.freq_num)]
    
    def get_times(self):
        times_inds = self.get_inds(self.TIME_IND)
        return self.data[:, times_inds]
    
    def get_temperatures(self):
        temperature_inds = self.get_inds(self.TMPA_IND)
        return self.data[:, temperature_inds]
    
    def get_capacitances(self):
        capacitance_inds = self.get_inds(self.CAPT_IND)
        return self.data[:, capacitance_inds]
    
    def get_losses(self):
        loss_inds = self.get_inds(self.LOSS_IND)
        return self.data[:, loss_inds]
    
    def time_derivative_filter(self):
        time_inds = self.get_inds(self.TIME_IND)
        temp_inds = self.get_inds(self.TMPA_IND)
        derivative =  (self.data[1:, temp_inds] - self.data[:-1, temp_inds]) / (self.data[1:, time_inds] - self.data[:-1, time_inds])

        not_flat = np.ones(derivative.shape, dtype=bool)
        not_flat[np.where(np.isclose(derivative, 0))] = False
        not_flat_inds = list(np.where(not_flat.all(axis=1))[0])
        
        self.data = self.data[not_flat_inds, :]
    
    def remove_bad_data(self, cap_cut = 0.5):
        caps = self.get_capacitances()
        bad = np.zeros(caps.shape, dtype=bool)
        bad[np.where(caps <= cap_cut)] = True
        bad_inds = np.any(bad, axis=1)
        
        self.data = self.data[np.logical_not(bad_inds), :]

    def temperature_range(self, start=None, end=None):
        temps = self.get_temperatures()
        in_range = np.ones(temps.shape, dtype=bool)
        if start is not None:
            in_range[np.where(temps < start)] = False
        if end is not None:
            in_range[np.where(temps > end)] = False
        inds = np.all(in_range, axis=1)

        self.data = self.data[inds, :]
    
    @staticmethod
    def loadtxt(filename: str) -> np.ndarray:
        return np.loadtxt(filename, comments="#", delimiter=",", skiprows=3, dtype=np.float64)
    
    def show(self):
        if self.freq_num == 3:
            colors = self.COLORS[::3]
        else:
            colors = self.COLORS
        if self.reverse:
            colors = colors[::-1]
        fig, (axre, axim) = plt.subplots(2, 1, figsize=(3.5, 6.5))
        axre.grid(linestyle='dotted')
        axim.grid(linestyle='dotted')
        for ii in range(self.freq_num):
            if self.reverse:
                ii = self.freq_num - ii - 1
            freq_name = str(int(self.frequencies[ii]))
            if len(freq_name) > 3:
                freq_name = freq_name[:-3] + " kHz"
            else:
                freq_name += " Hz"
            axre.scatter(
                self.data[:, self.TMPA_IND + ii * self.COLS_PER],
                self.data[:, self.CAPT_IND + ii * self.COLS_PER],
                s=4,
                marker="o",
                edgecolors=colors[ii],
                lw=.75,
                alpha=1,
                facecolor='w',
                label=freq_name
            )
            axim.scatter(
                self.data[:, self.TMPA_IND + ii * self.COLS_PER],
                self.data[:, self.LOSS_IND + ii * self.COLS_PER],
                s=4,
                marker="o",
                edgecolors=colors[ii],
                lw=.75,
                alpha=1,
                facecolor='w'
            )
        axre.set_title(self.name)
        axim.set_xlabel("Temperature (K)")
        axre.set_ylabel("Capacitance (pF)")
        axim.set_ylabel("Loss Tangent")
        axre.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        axim.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        axre.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        axim.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        axre.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        axim.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        fig.tight_layout()
        return fig, (axre, axim)


class DataSetOld(DataSet):
    TIME_IND = 0
    TMPA_IND = 1
    TMPB_IND = 2
    FREQ_IND = 3
    CAPT_IND = 4
    LOSS_IND = 5
    COLS_PER = 6


def reverse_freqs_in_data_set(file: Path):
    data_set = DataSet(file)
    data = data_set.data
    new_data = np.empty(data.shape)
    for ii in range(data_set.freq_num):
        start_placement = ii * DataSet.COLS_PER
        end_placenemt = start_placement + DataSet.COLS_PER
        end_grab = (data_set.freq_num - ii) * DataSet.COLS_PER
        start_grab = end_grab - DataSet.COLS_PER
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
    figs = [None] * 7
    axs = [None] * 7
    frequencies = (300, 700, 1400, 3000, 7000, 14000, 20000)
    for ii in range(7):
        figs[ii], axs[ii] = plt.subplots(1, 1)
        axs[ii].set_title(f"Frequency {frequencies[ii]} Hz")
        axs[ii].set_xlabel("Temperature (K)")
        axs[ii].set_ylabel("Capacitance (pF)")
        
    dir = Path.cwd()
    ii = 0
    for dir_ in dir.iterdir():
        for file in dir_.glob("*.csv"):
            data = DataSet(file)
            # print(data.data)
            print(data.shape)
            print(data.frequencies)
            time = data.get_times()
            temp = data.get_temperatures()
            caps = data.get_capacitances()
            loss = data.get_losses()
            start = 100
            #axs[ii].set_title(file.name)
            for ff in range(data.freq_num):
                axs[ff].plot(temp[start:, ff], caps[start:, ff], label=file.name)
            #plt.figure()
            #plt.title(file.name)
            #for ii in range(data.freq_num):
            #    plt.plot(temp[start:, ii], loss[start:, ii])
            #plt.figure()
            #plt.title(file.name)
            #for ii in range(data.freq_num):
            #    plt.plot(time[start:, ii], temp[start:, ii])
            ii += 1

    for ii in range(7):
        axs[ii].legend()
        figs[ii].tight_layout()
    plt.show()
    