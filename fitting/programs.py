from fitting.dielectric.bare import Bare
from fitting.dielectric.data import RawData, ProcessedFile, ProcessedFileLite, ProcessedPowder
from fitting.dielectric.calibrate import Calibrate
from fitting.dielectric.powder import Powder
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pylab as plt
import argparse
import toml

plt.style.use("fitting.style")


def bare_fit():
    parser = argparse.ArgumentParser(
        prog="bare-fit",
        description="Fit a bare capacitance file",
        epilog="author: Teddy Tortorici <edward.tortorici@colorado.edu>"
    )
    parser.add_argument("file", help="The CSV file containing bare capacitance data")
    parser.add_argument("real_order", type=int, help="The polynomial order for fitting the capacitance data")
    parser.add_argument("imaginary_order", type=int, help="The polynomial order for fitting the loss tangent data")
    parser.add_argument("-M", "--max_temperature", type=float, default=None, help="Optionally cut out temperatures above this value (in K).")
    parser.add_argument("-F", "--no_peaks", action="store_true", help="Don't fit peaks in the loss")
    parser.add_argument("-C", "--peak_center", type=float, default=None, help="Lock peak center to a specific temperature")
    args = parser.parse_args()

    files = [Path(f).resolve() for f in args.file.split(",")]

    bare = Bare(files, args.max_temperature, args.peak_center)
    bare.fit(real_order=args.real_order, imag_order=args.imaginary_order, peaks=not args.no_peaks)
    
    keys_real = [f"order 0 @ {int(f)} Hz" for f in bare.freq] + [f"order {ii}" for ii in range(1, args.real_order + 1)]
    keys_imag = [f"order 0 @ {int(f)} Hz" for f in bare.freq] 
    if not args.no_peaks:
        keys_imag += [f"height @ {int(f)} Hz" for f in bare.freq] + [f"width @ {int(f)} Hz" for f in bare.freq] 
    keys_imag += [f"order {ii}" for ii in range(1, args.imaginary_order + 1)]
    
    rslt_real = [float(rslt) for rslt in bare._fit_real]
    rslt_imag = [float(rslt) for rslt in bare._fit_imag]
    results_real = dict(zip(keys_real, rslt_real))
    results_imag = dict(zip(keys_imag, rslt_imag))
    results = {
        "capacitance": results_real,
        "loss tangent": results_imag,
    }

    now = datetime.now()

    with open(f"bare-fit-{now.year}-{now.month:02}-{now.day:02}_{now.hour:02}-{now.minute:02}.toml", "w") as toml_file:
        toml.dump(results, toml_file)
    
    plt.show()
    

def calibrate_capacitor():
    parser = argparse.ArgumentParser(
        prog="calibrate-capacitor",
        description="Process a calibrated data set with real and imaginary dielectric constant",
        epilog="author: Teddy Tortorici <edward.tortorici@colorado.edu>"
    )
    parser.add_argument("bare_file", help="The CSV file containing bare capacitance data.")
    parser.add_argument("film_file", help="The CSV file containing film capacitance measurement data.")
    parser.add_argument("real_order", type=int, help="The polynomial order for fitting the capacitance data.")
    parser.add_argument("imaginary_order", type=int, help="The polynomial order for fitting the loss tangent data.")
    parser.add_argument("film_thickness", type=float, help="Thickness of film in nanometers.")
    parser.add_argument("gap_width", type=float, help="gap width in microns.")
    parser.add_argument("-N", "--finger_num", type=int, default=50, help="Number of fingers on the capacitor.")
    # parser.add_argument("-TE", "--thickness_error", type=float, default=0., help="Estimated film thickness error in nanometers.")
    # parser.add_argument("-GE", "--gap_error", type=float, default=0., help="Estimated error of the gap width in microns.")
    # parser.add_argument("-FE", "--finger_length_error", type=float, default=0., help="Experimental error of the finger length in microns (should be roughly half the over-etching).")
    parser.add_argument("-MF", "--max_temperature_fit", type=float, help="Cut off temperatures above this value (in K).")
    parser.add_argument("-MD", "--max_temperature_data", type=float, help="Cut off temperatures in Lite file (in K)")
    parser.add_argument("-F", "--no_peaks", action="store_true", help="Don't fit peaks in the loss")
    args = parser.parse_args()

    args_dict = vars(args)

    now = datetime.now()

    with open(f"calibration-{now.year}-{now.month:02}-{now.day:02}_{now.hour:02}-{now.minute:02}.toml", "w") as toml_file:
        toml.dump(args_dict, toml_file)

    bare_files = [Path(f).resolve() for f in args.bare_file.split(",")]
    film_files = [Path(f).resolve() for f in args.film_file.split(",")]

    cal = Calibrate(film_files)
    cal.load_calibration(bare_files, args.real_order, args.imaginary_order,
                         peaks=not args.no_peaks,
                         max_temperature_fit=args.max_temperature_fit)
    cal.run(args.film_thickness * 1e-9,
            args.gap_width * 1e-6,
            finger_num=args.finger_num,
            # gap_err=args.gap_error * 1e-6,
            # film_thickness_err=args.thickness_error * 1e-9,
            # finger_length_err=args.finger_length_error * 1e-6,
            max_temperature_data=args.max_temperature_data)
    

def process_powder():
    parser = argparse.ArgumentParser(
        prog="process-powder",
        description="Process powder data",
        epilog="author: Teddy Tortorici <edward.tortorici@colorado.edu>"
    )
    parser.add_argument("powder_file", help="The CSV file containing powder capacitance measurement data.")
    parser.add_argument("-B", "--bare", help="Measured bare capacitance at room temperature.")
    parser.add_argument("-1", "--linear", type=float, default=2.795173e-05, help="Linear dependence of the capacitance.")
    parser.add_argument("-2", "--quadratic", type=float, default=1.141850e-07, help="Quadratic dependence of the capacitance.")
    parser.add_argument("-3", "--quartic", type=float, default=2.817504e-10, help="Quartic dependence of the capacitance.")
    parser.add_argument("-epss", "--substrate_epsilon", type=float, default=3.8, help="Dielectric constant of the silicon substrate.")
    parser.add_argument("-MD", "--max_temperature_data", type=float, help="Cut off temperatures in Lite file (in K).")
    parser.add_argument("-S", "--sorted", action="store_true", help="Use this flag if the data is already sorted (unique columns for each frequency).")
    args = parser.parse_args()

    args_dict = vars(args)

    now = datetime.now()

    with open(f"powder-process-{now.year}-{now.month:02}-{now.day:02}_{now.hour:02}-{now.minute:02}.toml", "w") as toml_file:
        toml.dump(args_dict, toml_file)

    powder_files = [Path(f).resolve() for f in args.powder_file.split(",")]

    if args.bare is not None:
        args.bare = np.array(args.bare.split(","), dtype=np.float64)

    pow = Powder(data_files=powder_files,
                 room_temperature_capacitance=args.bare,
                 linear_term=args.linear,
                 quadratic_term=args.quadratic,
                 quartic_term=args.quartic,
                 epsilon_substrate=args.substrate_epsilon,
                 already_sorted=args.sorted)
    pow.run(max_temperature_data=args.max_temperature_data)


def plot():
    parser = argparse.ArgumentParser(
        prog="plot-spectra",
        description="Plot dielectric spectroscopy data.",
        epilog="author: Teddy Tortorici <edward.tortorici@colorado.edu>"
    )
    parser.add_argument("file_name", help="File, or comma-separated files, to load and plot.")
    parser.add_argument("-RL", "--real_limits", help="ylim for real part.")
    parser.add_argument("-IL", "--imaginary_limits", help="ylim for imaginary part.")
    parser.add_argument("-TL", "--temperature_limit", type=float, default=None, help="Cut data below this temperature.")
    parser.add_argument("-S", "--save", help="Save with specified filename")
    parser.add_argument("-DPI", "--dpi", type=int, default=300, help="Change DPI for saving image.")
    args = parser.parse_args()

    files = [Path(f).resolve() for f in args.file_name.split(",")]

    if files[0].stem.endswith("lite"):
        data = ProcessedFileLite(files)
    elif files[0].stem.endswith("calibration"):
        data = ProcessedFile(files)
    elif files[0].stem.endswith("powder-process"):
        data = ProcessedPowder(files)
    else:
        data = RawData(files)
    
    if args.temperature_limit is not None:
        data.set_temperature_cut(args.temperature_limit)

    fig, (ax_re, ax_im) = data.plot()

    if args.real_limits is not None:
        ax_re.set_ylim([float(num) for num in args.real_limits.split(",")])
    if args.imaginary_limits is not None:
        ax_im.set_ylim([float(num) for num in args.imaginary_limits.split(",")])

    if args.save is not None:
        fig.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
    plt.show()
