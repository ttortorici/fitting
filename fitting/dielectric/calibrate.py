from fitting.dielectric.bare import Bare
from fitting.dielectric.data import RawData, ProcessedFile, ProcessedFileLite
from fitting.dielectric.variance import calculate as determine_variance
import fitting.capacitor as capacitor
from pathlib import Path
import numpy as np


class Calibrate:

    def __init__(self, film_data_file: Path):
        if isinstance(film_data_file, Path):
            self.dir = film_data_file.parent
            self.name = film_data_file.stem
        else:
            self.dir = film_data_file[0].parent
            self.name = film_data_file[0].stem
        self.bare_cap_curve = None      # will be a 2D array of size (data_points, frequency_num)
        self.bare_loss_curve = None     # will be a 2D array of size (data_points, frequency_num)
        self.bare_cap_std = None        # will be a 1D array of length frequency_num
        self.bare_loss_std = None       # will be a 1D array of length frequency_num
        self.raw_data = RawData(film_data_file)
        print()

    def load_calibration(self, file: Path, real_poly_order: int, imaginary_poly_order: int,
                         peaks: bool, max_temperature_fit: float):
        bare = Bare(file, max_temperature_fit)
        bare.fit(real_order=real_poly_order, imag_order=imaginary_poly_order, peaks=peaks)
        
        temperature = self.raw_data.get_temperatures()
        self.bare_cap_curve = bare.fit_function_real(bare._fit_real, temperature)
        if peaks:
            self.bare_loss_curve = bare.fit_function_imag_peaks(bare._fit_imag, temperature)
        else:
            self.bare_loss_curve = bare.fit_function_imag_poly(bare._fit_imag, temperature)
        if self.bare_loss_curve.shape[0] == 1:
            self.bare_loss_curve = np.tile(self.bare_loss_curve, (temperature.shape[0], 1))
        self.bare_cap_dev = bare.standard_dev_real
        self.bare_loss_dev = bare.standard_dev_imag

    def run(self, film_thickness: float, gap_width: float, finger_num: int=50,
            max_temperature_data: float=None):
        if max_temperature_data is not None:
            temperature_mask = np.all(self.raw_data.get_temperatures() < max_temperature_data, axis=1)
        else:
            temperature_mask = np.ones(self.raw_data.shape[0], dtype=bool)

        data_pts = self.raw_data.shape[0]       # number of rows
        print(f"num data: {data_pts}")
        time = self.raw_data.get_times()
        temp = self.raw_data.get_temperatures()
        tempb = self.raw_data.get_shield_temperatures()
        cap_real = self.raw_data.get_capacitances()

        loss = self.raw_data.get_losses()
        volt = self.raw_data.get_voltages()
        cap_imag = cap_real * loss
        cap_bare_imag = self.bare_cap_curve * self.bare_loss_curve
        cap_real_dev, cap_imag_dev = determine_variance(temp, cap_real, cap_imag) 
        
        del_cap_real = cap_real - self.bare_cap_curve   # pF
        del_cap_imag = cap_imag - cap_bare_imag         # pF
        del_cap_real_dev, del_cap_imag_dev = determine_variance(temp, del_cap_real, del_cap_imag)

        eps_real, eps_imag = capacitor.dielectric_constant(
            del_cap_real, del_cap_imag, gap_width, film_thickness,
            finger_length=1e-3, finger_num=finger_num
        )

        eps_real_dev, eps_imag_dev = determine_variance(temp, eps_real, eps_imag)

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
                cap_real[:, ff],                            # C'
                cap_real_dev[:, ff],
                self.bare_cap_curve[:, ff],                 # C'_b
                del_cap_real[:, ff],                        # del C'          
                del_cap_real_dev[:, ff],
                loss[:, ff],                                # tan(delta)
                self.bare_loss_curve[:, ff],                # tan(delta)_b
                cap_imag[:, ff],                            # C''
                cap_imag_dev[:, ff],
                del_cap_imag[:, ff],                        # del C''
                del_cap_imag_dev[:, ff],
                eps_real[:, ff],                            # real dielectric constant
                eps_real_dev[:, ff],
                eps_imag[:, ff],                            # imaginary dielectric constant
                eps_imag_dev[:, ff],
                volt[:, ff],
                np.ones(data_pts) * freq,
            ), axis=1)
            data_at_freq_lite[ff] = np.stack((
                time[temperature_mask][:, ff],
                temp[temperature_mask][:, ff],
                cap_real[temperature_mask][:, ff] - self.bare_cap_curve[temperature_mask][:, ff],   # del C'          
                del_cap_real_dev[temperature_mask][:, ff],
                del_cap_imag[temperature_mask][:, ff],                        # del C''
                del_cap_imag_dev[temperature_mask][:, ff],
                eps_real[temperature_mask][:, ff],                            # real dielectric constant
                eps_real_dev[temperature_mask][:, ff],
                eps_imag[temperature_mask][:, ff],                            # imaginary dielectric constant
                eps_imag_dev[temperature_mask][:, ff],
                volt[temperature_mask][:, ff],
                np.ones(np.sum(temperature_mask)) * freq,
            ), axis=1)
        data = np.hstack(data_at_freq)
        data_lite = np.hstack(data_at_freq_lite)
        labels = ", ".join(labels)
        labels_lite = ", ".join(labels_lite)
        np.savetxt(self.dir / (self.name + "__calibrated.csv"), data, delimiter = ", ", header=labels)
        np.savetxt(self.dir / (self.name + "__calibrated_lite.csv"), data_lite, delimiter = ", ", header=labels_lite)


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

    # dir = Path("C:\\Users\\Teddy\\OneDrive - UCB-O365\\Rogerslab3\\Teddy\\TPP Films\\BTB-TPP\\2024 Film Growth\\Film 2\\BDS\\00 - Cals")
    # file = dir / "2024-02-29__none__M09-1-TT501__CAL__T-13-39_rev-freq.csv"

    """BAP Film 2"""
    # path = Path(r"C:\Users\Teddy\OneDrive - UCB-O365\Rogerslab3\Teddy\TPP Films\BAP-TPP\Film 20241014\Dielectric")
    # bare_file = "2024-10-11__bare__M01-2__CAL__T-12-16.csv"
    # film_file = "2024-10-15__BAPaTPP2-27C__M01-2__FILM__T-08-58.csv"
    # file = Path(r"H:\OneDrive - UCB-O365\Rogerslab3\Teddy\TPP Films\BAP-TPP\Film 20241014\Dielectric\2024-10-11__bare__M01-2__CAL__T-12-16.csv")
    
    """OLD BTP Film 2017"""
    path = Path(r"C:\Users\Teddy\OneDrive - UCB-O365\Rogerslab3\Teddy\TPP Films\Old films\2017\11 - Novemember\BTP\BDS")
    bare_file = "Calibration_TT2-17-15_Bare_1510274316_07_sorted_rev-freq.csv"
    film_file1 = "HeliumCool_TT2-17-15_film_WYC103_1510686381_3_sorted_rev-freq.csv"
    film_file2 = "HeliumCool_TT2-17-15_film_WYC103_1510694868_27_sorted_rev-freq.csv"
    cal = Calibrate((path / film_file1, path / film_file2))
    cal.load_calibration(path / bare_file, 5, 2)
    cal.run()
