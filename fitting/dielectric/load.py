from pathlib import Path
import numpy as np
from fitting.dielectric.calibrate import

class DataFile:
    TIME_IND = 0
    TMPA_IND = 1
    TMPB_IND = 2
    CAPT_IND = 3
    LOSS_IND = 4
    VOLT_IND = 5
    FREQ_IND = 6
    COLS_PER = 7
    BCAP_IND = 7
    BLOS_IND = 8
    DCRE_IND = 9
    DCIM_IND = 10
    COLS_PRO = 11

    def __init__(self, files: Path):
        if isinstance(files, Path) or isinstance(files, str):
            self.data = self.loadtxt(files)
        elif isinstance(files, list) or isinstance(files, tuple):
            self.data = self.loadtxt(files[0])
            for file in files[1:]:
                self.data = np.append(self.data, self.loadtxt(file), axis=0)
        self.shape = self.data.shape

        self.freq_num = int(self.shape[1] / self.COLS_PER)
        self.freqs = [self.data[0, index] for index in self.get_inds(self.FREQ_IND)]
    
    def get_inds(self, col_ind):
        return [col_ind + DataSet.COLS_PER * ii for ii in range(self.freq_num)]
    
    def get_times(self):
        times_inds = self.get_inds(DataSet.TIME_IND)
        return self.data[:, times_inds]
    
    def get_temperatures(self):
        temperature_inds = self.get_inds(DataSet.TMPA_IND)
        return self.data[:, temperature_inds]
    
    def get_capacitances(self):
        capacitance_inds = self.get_inds(DataSet.CAPT_IND)
        return self.data[:, capacitance_inds]
    
    def get_losses(self):
        loss_inds = self.get_inds(DataSet.LOSS_IND)
        return self.data[:, loss_inds]

    @staticmethod
    def loadtxt(filename: str) -> np.ndarray:
        return np.loadtxt(filename, comments="#", delimiter=",", skiprows=3, dtype=np.float64)


class DataSet(DataFile):

    def __init__(self, files):
        super().__init__(files)
        self.time_derivative_filter()
    
    def time_derivative_filter(self):
        time_inds = self.get_inds(DataSet.TIME_IND)
        temp_inds = self.get_inds(DataSet.TMPA_IND)
        derivative =  (self.data[1:, temp_inds] - self.data[:-1, temp_inds]) / (self.data[1:, time_inds] - self.data[:-1, time_inds])

        not_flat = np.ones(derivative.shape, dtype=bool)
        not_flat[np.where(np.isclose(derivative, 0))] = False
        not_flat_inds = list(np.where(not_flat.all(axis=1))[0])
        
        self.data = self.data[not_flat_inds, :]


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

    file = Path(r"C:\Users\Teddy\OneDrive - UCB-O365\Rogerslab3\Teddy\TPP Films\BAP-TPP\Film 20241014\Dielectric\2024-10-11__bare__M01-2__CAL__T-12-16.csv")
    data = DataFile(file)

    # dir = Path("C:\\Users\\Teddy\\OneDrive - UCB-O365\\Rogerslab3\\Teddy\\TPP Films\\BTB-TPP\\2024 Film Growth\\Film 2\\BDS\\00 - Cals")
    # file = dir / "2024-02-29__none__M09-1-TT501__CAL__T-13-39_rev-freq.csv"
    # data = DataSet(file)
    # # print(data.data)
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
    
