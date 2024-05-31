# fitting
 
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
