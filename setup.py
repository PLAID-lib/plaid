# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#

"""Configuration for building and installing the plaid package."""

import os
import subprocess

from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext

HDF5_VERSION = "1.14.6"


def build_hdf5():
    """HDF5 packaging."""
    tmpdir = os.path.abspath("build_hdf5")
    prefix = os.path.join(tmpdir, "install")
    os.makedirs(prefix, exist_ok=True)
    tarfile = f"hdf5-{HDF5_VERSION}.tar.gz"
    url = f"https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-{HDF5_VERSION[:4]}/hdf5-{HDF5_VERSION}/src/{tarfile}"

    # Download and extract
    subprocess.check_call(["curl", "-L", "-o", tarfile, url])
    subprocess.check_call(["tar", "xzf", tarfile])

    src_dir = f"hdf5-{HDF5_VERSION}"
    subprocess.check_call(
        [
            "./configure",
            f"--prefix={prefix}",
            "--enable-static",
            "--disable-shared",
            "--with-pic",
        ],
        cwd=src_dir,
    )
    subprocess.check_call(["make", "-j4"], cwd=src_dir)
    subprocess.check_call(["make", "install"], cwd=src_dir)

    # Environment for pyCGNS build
    os.environ["CFLAGS"] = f"-I{prefix}/include"
    os.environ["LDFLAGS"] = f"-L{prefix}/lib"
    os.environ["PKG_CONFIG_PATH"] = f"{prefix}/lib/pkgconfig"


class build_ext(_build_ext):
    """Build class."""

    def run(self):
        """Build run function."""
        build_hdf5()
        super().run()


setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    cmdclass={"build_ext": build_ext},
)
