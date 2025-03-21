import os
from glob import glob

from setuptools import find_packages, setup

package_name = "ft_calibration"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Chen Chen",
    maintainer_email="maker_cc@foxmail.com",
    description="Force torque sensor calibration for ROS 2",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "ft_sampler_node = ft_calibration.ft_sampler_node:main",
            "ft_calibration_gui = ft_calibration.ft_calibration_gui:main",
        ],
    },
)
