from .bare import Bare
from .data import RawData, ProcessedFile, ProcessedFileLite
from .calibrate import Calibrate
from .histogram import LoadFit as Histogram
from .variance import calculate as variance
import matplotlib.pylab as plt
plt.style.use("fitting.style")