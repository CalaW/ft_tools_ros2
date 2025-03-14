import os
import subprocess
from pathlib import Path

cwd = Path(__file__).resolve().parent
os.chdir(cwd)
# for each .ui file run a command
for file in Path().glob("*.ui"):
    py_file = cwd.parent / "ft_calibration" / file.with_suffix(".py").name
    # generate python file from .ui file
    subprocess.run(["pyuic5", file, "-o", py_file], check=True)
