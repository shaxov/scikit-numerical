import numpy
import setuptools
from distutils.core import Extension

with open("README.md", "r") as fh:
    long_description = fh.read()

_interpolation_module = Extension('_interpolation', sources=['numerical/_interpolation.c'],
                                  include_dirs=[numpy.get_include()],
                                  extra_compile_args=[])


setuptools.setup(
    name="scikit-numerical",
    version="0.1.1",
    author="Maksym Shpakovych",
    author_email="maksym.shpakovych@gmail.com",
    description="Tools for numerical math calculations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Bellator95/scikit-numerical",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    ext_modules=[_interpolation_module],
)
