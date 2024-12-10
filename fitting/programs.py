from fitting.dielectric.bare import Bare
from fitting.dielectric.load import RawData, ProcessedFile, ProcessedFileLite
from fitting.dielectric.calibrate import Calibrate
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
    parser.add_argument("-M", "--max-temperature", type=float, default=None, help="Optionally cut out temperatures above this value (in K).")
    args = parser.parse_args()

    files = [Path(f).resolve() for f in args.file.split(",")]

    bare = Bare(files, args.max_temperature)
    bare.fit(real_order=args.real_order, imag_order=args.imaginary_order)
    
    keys_real = [f"order 0 @ {int(f)} Hz" for f in bare.freq] + [f"order {ii}" for ii in range(1, args.real_order + 1)]
    keys_imag = [f"order 0 @ {int(f)} Hz" for f in bare.freq] + [f"height @ {int(f)} Hz" for f in bare.freq] + [f"width @ {int(f)} Hz" for f in bare.freq] + [f"order {ii}" for ii in range(1, args.imaginary_order + 1)]
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
        description="Process a calibrated data set with real and imaginary electric susceptibility",
        epilog="author: Teddy Tortorici <edward.tortorici@colorado.edu"
    )
    parser.add_argument("bare_file", help="The CSV file containing bare capacitance data.")
    parser.add_argument("film_file", help="The CSV file containing film capacitance measurement data.")
    parser.add_argument("real_order", type=int, help="The polynomial order for fitting the capacitance data.")
    parser.add_argument("imaginary_order", type=int, help="The polynomial order for fitting the loss tangent data.")
    parser.add_argument("film_thickness", type=float, help="Thickness of film in nanometers.")
    parser.add_argument("gap_width", type=float, help="gap width in microns.")
    parser.add_argument("-N", "--finger_num", type=int, default=50, help="Number of fingers on the capacitor.")
    parser.add_argument("-TE", "--thickness_error", type=float, default=5., help="Estimated film thickness error in nanometers.")
    parser.add_argument("-GE", "--gap_error", type=float, default=0.2, help="Estimated error of the gap width in microns.")
    parser.add_argument("-FE", "--finger_length_error", type=float, default=.5, help="Experimental error of the finger length in microns (should be roughly half the over-etching).")
    parser.add_argument("-M", "--max_temperature", help="Cut off temperatures above this value (in K).")
    parser.add_argument("-pp", "--parallel_plate", action="store_true", help="Use parallel plate approximation.")
    args = parser.parse_args()

    args_dict = vars(args)

    now = datetime.now()

    with open(f"calibration-{now.year}-{now.month:02}-{now.day:02}_{now.hour:02}-{now.minute:02}.toml", "w") as toml_file:
        toml.dump(args_dict, toml_file)

    bare_files = [Path(f).resolve() for f in args.bare_file.split(",")]
    film_files = [Path(f).resolve() for f in args.film_file.split(",")]

    cal = Calibrate(film_files)
    cal.load_calibration(bare_files, args.real_order, args.imaginary_order)
    cal.run(args.film_thickness * 1e-9,
            args.gap_width * 1e-6,
            finger_num=args.finger_num,
            gap_err=args.gap_error * 1e-6,
            film_thickness_err=args.thickness_error * 1e-9,
            finger_length_err=args.finger_length_error * 1e-6,
            parallel_plate=args.parallel_plate)


def plot():
    parser = argparse.ArgumentParser(
        prog="plot-spectra",
        description="Plot dielectric spectroscopy data.",
        epilog="author: Teddy Tortorici <edward.tortorici@colorado.edu"
    )
    parser.add_argument("file_name", help="File, or comma-separated files, to load and plot.")
    parser.add_argument("-RL", "--real_limits", help="ylim for real part.")
    parser.add_argument("-IL", "--imaginary_limits", help="ylim for imaginary part.")
    parser.add_argument("-TL", "--temperature_limit", type=float, default=None, help="Cut data below this temperature.")
    parser.add_argument("-S", "--save", help="Save with specified filename")
    parser.add_argument("-DPI", "--dpi", type=int, default=100, help="Change DPI for saving image.")
    args = parser.parse_args()

    files = [Path(f).resolve() for f in args.file_name.split(",")]

    if files[0].stem.endswith("lite"):
        data = ProcessedFileLite(files)
    elif files[0].stem.endswith("calibration"):
        data = ProcessedFile(files)
    else:
        data = RawData(files)
    
    fig, (ax_re, ax_im) = data.plot()

    if args.real_limits is not None:
        ax_re.set_ylim([float(num) for num in args.real_limits.split(",")])
    if args.imaginary_limits is not None:
        ax_im.set_ylim([float(num) for num in args.imaginary_limits.split(",")])

    if args.save is not None:
        fig.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
    plt.show()
