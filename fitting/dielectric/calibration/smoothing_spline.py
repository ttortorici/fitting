from scipy import interpolate
from load_data import DataSet
import numpy as np


def smoothing_spline(x, data):
    tck = interpolate.splrep(data[:, 0], data[:, 1], s=len(data))
    return interpolate.BSpline(*tck)(x)

if __name__ == "__main__":
    from pathlib import Path
    import matplotlib.pylab as plt

    dir = Path("C:\\Users\\Teddy\\OneDrive - UCB-O365\\Rogerslab3\\Teddy\\TPP Films\\BTB-TPP\\2024 Film Growth\\Film 2\\BDS\\00 - Cals")
    file = dir / "2024-02-29__none__M09-1-TT501__CAL__T-13-39_rev-freq.csv"
    
    start_clip = 300
    end_clip = 200

    dataset = DataSet(file)
    if end_clip:
        time = dataset.get_times()[start_clip:-end_clip, :]
        temp = dataset.get_temperatures()[start_clip:-end_clip, :]
        caps = dataset.get_capacitances()[start_clip:-end_clip, :]
        loss = dataset.get_losses()[start_clip:-end_clip, :]
    else:
        time = dataset.get_times()[start_clip:, :]
        temp = dataset.get_temperatures()[start_clip:, :]
        caps = dataset.get_capacitances()[start_clip:, :]
        loss = dataset.get_losses()[start_clip:, :]
    
    sorted_indices = np.argsort(temp[:, -1])
    sorted_array1 = temp[:, -1][sorted_indices]
    sorted_array2 = caps[:, -1][sorted_indices]

    print(temp[:, -1].shape)
    data = np.dstack((sorted_array1, sorted_array2))[0]
    print(data.shape)

    temp_spl = interpolate.UnivariateSpline(sorted_array1, sorted_array2, k=3)
    # tck = interpolate.splrep(sorted_array1, sorted_array2, s=.1)

    x = np.linspace(0, 400, 1000)
    y = temp_spl(x)
    # y = interpolate.splev(x, tck, der=0)
    # y = interpolate.BSpline(*tck)(x)
    print(y.shape)
    print(y.max())
    print(y.min())

    plt.figure()
    plt.plot(data[:, 0], data[:, 1], 'b')
    plt.plot(x, y, 'r-')
    
    plt.show()