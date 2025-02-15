# fitting

# Installation

## Set up new virtual environment

Create a virtual environment in order to isolate the CLI programs from the primary system:

### conda

```
conda create -n [preferred-venv-name] python=3.13
conda activate [preferred-venv-name]
```

### venv

```
python -m venv [path/to/venv/venv-name]
[path/to/venv/venv-name]/Scripts/Acitvate.ps1
```

## Clone and install

Clone the repository and then `cd` into the repository using the command prompt (or Anaconda prompt if using conda).

```
python -m pip install -U pip
pip install -U build
python -m build
pip install .\dist\fitting-2.4-py3-none-any.whl
```

(note, the 2.4 is the version number, and may differ from the README).

# Programs and notebook support

The following CLI programs are callable from the command prompt (as long as the virtual environment you installed the package in is activated)

`bare-fit`

```
usage: bare-fit [-h] [-M MAX_TEMPERATURE] [-F] file real_order imaginary_order

Fit a bare capacitance file

positional arguments:
  file                  The CSV file containing bare capacitance data
  real_order            The polynomial order for fitting the capacitance data
  imaginary_order       The polynomial order for fitting the loss tangent data

options:
  -h, --help            show this help message and exit
  -M, --max_temperature MAX_TEMPERATURE
                        Optionally cut out temperatures above this value (in K).
  -F, --no_peaks        Don't fit peaks in the loss

author: Teddy Tortorici <edward.tortorici@colorado.edu>
```

`calibrate-capacitor`

```
usage: calibrate-capacitor [-h] [-N FINGER_NUM] [-MF MAX_TEMPERATURE_FIT] [-MD MAX_TEMPERATURE_DATA] [-F] bare_file film_file real_order imaginary_order film_thickness gap_width

Process a calibrated data set with real and imaginary dielectric constant

positional arguments:
  bare_file             The CSV file containing bare capacitance data.
  film_file             The CSV file containing film capacitance measurement data.
  real_order            The polynomial order for fitting the capacitance data.
  imaginary_order       The polynomial order for fitting the loss tangent data.
  film_thickness        Thickness of film in nanometers.
  gap_width             gap width in microns.

options:
  -h, --help            show this help message and exit
  -N, --finger_num FINGER_NUM
                        Number of fingers on the capacitor.
  -MF, --max_temperature_fit MAX_TEMPERATURE_FIT
                        Cut off temperatures above this value (in K).
  -MD, --max_temperature_data MAX_TEMPERATURE_DATA
                        Cut off temperatures in Lite file (in K)
  -F, --no_peaks        Don't fit peaks in the loss

author: Teddy Tortorici <edward.tortorici@colorado.edu>
```

`process-powder`

```
usage: process-powder [-h] [-B BARE] [-1 LINEAR] [-2 QUADRATIC] [-3 QUARTIC] [-epss SUBSTRATE_EPSILON] [-MD MAX_TEMPERATURE_DATA] [-S] powder_file

Process powder data

positional arguments:
  powder_file           The CSV file containing powder capacitance measurement data.

options:
  -h, --help            show this help message and exit
  -B, --bare BARE       Measured bare capacitance at room temperature.
  -1, --linear LINEAR   Linear dependence of the capacitance.
  -2, --quadratic QUADRATIC
                        Quadratic dependence of the capacitance.
  -3, --quartic QUARTIC
                        Quartic dependence of the capacitance.
  -epss, --substrate-epsilon SUBSTRATE_EPSILON
                        Dielectric constant of the silicon substrate.
  -MD, --max_temperature_data MAX_TEMPERATURE_DATA
                        Cut off temperatures in Lite file (in K).
  -S, --sorted          Use this flag if the data is already sorted (unique columns for each frequency).

author: Teddy Tortorici <edward.tortorici@colorado.edu>
```

`plot-spectra`

```
usage: plot-spectra [-h] [-RL REAL_LIMITS] [-IL IMAGINARY_LIMITS] [-TL TEMPERATURE_LIMIT] [-S SAVE] [-DPI DPI] file_name

Plot dielectric spectroscopy data.

positional arguments:
  file_name             File, or comma-separated files, to load and plot.

options:
  -h, --help            show this help message and exit
  -RL, --real_limits REAL_LIMITS
                        ylim for real part.
  -IL, --imaginary_limits IMAGINARY_LIMITS
                        ylim for imaginary part.
  -TL, --temperature_limit TEMPERATURE_LIMIT
                        Cut data below this temperature.
  -S, --save SAVE       Save with specified filename
  -DPI, --dpi DPI       Change DPI for saving image.

author: Teddy Tortorici <edward.tortorici@colorado.edu>
```

## X-ray fitting
Starting cell example

```Python
%matplotlib widget
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from fitting.xrd import Hexagonal, q_from_2th

filename = Path.cwd() / "GBA122.xy.TXT"
data = np.loadtxt(filename, delimiter=",")

hex = Hexagonal(11.8, 10, q_from_2th(data[:, 0]), data[:, 1], r"30%$\bf{2}$@TPP-d$_{12}$")
print(f"lowest y-axis value is {data[:, 1].min()}")
```
