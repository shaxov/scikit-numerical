import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scikit-numerical",
    version="0.0.8",
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
)
