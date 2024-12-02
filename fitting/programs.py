from fitting.dielectric.bare import Bare
from fitting.dielectric.load import RawData, ProcessedFile
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
    parser.add_argument("real_order", help="The polynomial order for fitting the capacitance data")
    parser.add_argument("imaginary_order", help="The polynomial order for fitting the loss tangent data")
    parser.add_argument("-M", "--max-temperature", help="Optionally cut out temperatures above this value (in K).")
    args = parser.parse_args()

    files = [Path(f).resolve() for f in args.file.split(",")]

    bare = Bare(files, args.max_temperature)
    bare.fit(real_order=int(args.real_order), imag_order=int(args.imaginary_order))
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
    parser.add_argument("-TE", "--thickness_error", type=float, default=10., help="Estimated film thickness error in nanometers.")
    parser.add_argument("-GE", "--gap_error", type=float, default=1., help="Estimated error of the gap width in microns.")
    parser.add_argument("-FE", "--finger_length_error", type=float, default=1., help="Experimental error of the finger length in microns (should be roughly half the over-etching).")
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
