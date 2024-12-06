import sys
from pathlib import Path

venv = Path(sys.prefix)
package = venv / "Lib/site-packages/fitting/"

config = package / "config.toml"